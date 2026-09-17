"""Validate and report all 42 fading fits, including paired comparisons."""

from __future__ import annotations

import csv
import gzip
import json

import numpy as np
from test_fading_failures import CASES, DOCS, ROOT, VARIANTS, signature

METRICS = ("pitch_mae_cents", "onset_mae_ms", "joint_event_error", "log_spectral_distance_db")
LABELS = {"1000hz": "1 s / 1000 Hz", "2oct": "1 s / 2 oct", "4oct": "1 s / 4 oct"}
EXPLORED = {"C04-T0005", "C08-T0000"}


def write_csv(name, rows):
    with (DOCS / name).open("w") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    results, rows = [], []
    for index in range(42):
        with gzip.open(ROOT / f"{index:02d}.json.gz", "rt") as handle:
            d = json.load(handle)
        assert d["signature"] == signature() and d["index"] == index
        assert d["case"]["target_id"] == CASES[index // 6]["target_id"]
        assert d["variant"] == VARIANTS[index % 3]
        assert d["mode"] == ("plateau" if index % 6 < 3 else "fresh")
        fit = d["fit"]
        assert fit["updates"] <= 3000
        assert abs(fit["best_loss"] - min(s["raw_loss"] for s in fit["trajectory"])) < 1e-12
        assert np.isfinite([d["metrics"][k] for k in METRICS]).all()
        row = dict(
            target_id=d["case"]["target_id"],
            events=d["case"]["events"],
            mode=d["mode"],
            variant=d["variant"],
            previously_explored=d["case"]["target_id"] in EXPLORED,
            **{k: d["metrics"][k] for k in METRICS},
            canonical_loss=d["canonical_loss"],
            updates=fit["updates"],
            stopped_by=fit["stopped_by"],
            wall_seconds=fit["wall_seconds"],
            recovered=d["metrics"]["pitch_mae_cents"] < 1 and d["metrics"]["onset_mae_ms"] < 1,
        )
        results.append(d)
        rows.append(row)
    lookup = {(r["target_id"], r["mode"], r["variant"]): r for r in rows}
    assert len(lookup) == 42
    for case in CASES:
        for mode in ("plateau", "fresh"):
            group = [
                d
                for d in results
                if d["case"]["target_id"] == case["target_id"] and d["mode"] == mode
            ]
            assert all(d["initial_metrics"] == group[0]["initial_metrics"] for d in group)
    write_csv("per_phrase.csv", rows)
    summary, paired = [], []
    for scope in ("all_seven", "other_five"):
        chosen = [r for r in rows if scope == "all_seven" or not r["previously_explored"]]
        for mode in ("plateau", "fresh"):
            for variant in VARIANTS:
                group = [r for r in chosen if r["mode"] == mode and r["variant"] == variant]
                for metric in (*METRICS, "updates", "wall_seconds"):
                    values = np.array([r[metric] for r in group])
                    summary.append(
                        dict(
                            scope=scope,
                            mode=mode,
                            variant=variant,
                            metric=metric,
                            n=len(values),
                            mean=float(values.mean()),
                            std=float(values.std(ddof=1)),
                            median=float(np.median(values)),
                            recovered=sum(r["recovered"] for r in group),
                        )
                    )
                if variant != "1000hz":
                    diff = np.array(
                        [
                            r["joint_event_error"]
                            - lookup[r["target_id"], mode, "1000hz"]["joint_event_error"]
                            for r in group
                        ]
                    )
                    paired.append(
                        dict(
                            scope=scope,
                            mode=mode,
                            variant=variant,
                            n=len(group),
                            better=int((diff < -1e-10).sum()),
                            worse=int((diff > 1e-10).sum()),
                            tied=int((np.abs(diff) <= 1e-10).sum()),
                            median_difference=float(np.median(diff)),
                        )
                    )
    write_csv("summary.csv", summary)
    write_csv("paired.csv", paired)
    with gzip.open(DOCS / "raw-results.json.gz", "wt") as handle:
        json.dump(results, handle, separators=(",", ":"))
    lines = [
        "# Fading across seven failed phrases",
        "",
        "Complete: 42 fits, three frequency horizons × two starting conditions × seven phrases.",
        "",
        "[Protocol](protocol.md) · [Per-phrase metrics](per_phrase.csv) · "
        "[Mean/SD/median](summary.csv) · [Paired comparisons](paired.csv) · "
        "[Full trajectories](raw-results.json.gz)",
        "",
        "All four diagonal directions; 1-second logarithmic fade in time; fixed amplitudes; "
        "no Log-Weighing. Same registered Adam/patience and 3000-update cap. "
        "Each fit selects its best own-objective iterate. "
        "Values below are pitch MAE (cents) / onset MAE (ms).",
        "",
    ]
    for mode in ("fresh", "plateau"):
        lines += [
            f"## {mode.title()} start",
            "",
            "| Case | 1 s / 1000 Hz | 1 s / 2 oct | 1 s / 4 oct |",
            "|---|---:|---:|---:|",
        ]
        for case in CASES:
            cid = case["target_id"]
            cells = [
                f"{lookup[cid, mode, v]['pitch_mae_cents']:.3f} / "
                f"{lookup[cid, mode, v]['onset_mae_ms']:.3f}"
                for v in VARIANTS
            ]
            lines.append(
                "| " + cid + (" *" if cid in EXPLORED else "") + " | " + " | ".join(cells) + " |"
            )
        lines += [""]
    lines += [
        "* C04-T0005 is the original development case; C08-T0000 was used to explore fading. "
        "Separate summaries exclude these two and retain the other five failures. "
        "The seven-case set is selected for failure, not representative of all targets.",
        "",
        "## Recovery counts",
        "",
        "Both pitch MAE <1 cent and onset MAE <1 ms.",
        "",
        "| Set | Start | 1000 Hz | 2 oct | 4 oct |",
        "|---|---|---:|---:|---:|",
    ]
    for scope in ("all_seven", "other_five"):
        for mode in ("fresh", "plateau"):
            counts = [
                next(
                    r["recovered"]
                    for r in summary
                    if r["scope"] == scope and r["mode"] == mode and r["variant"] == v
                )
                for v in VARIANTS
            ]
            lines.append("| " + scope + " | " + mode + " | " + " | ".join(map(str, counts)) + " |")
    lines += [
        "",
        "![Joint errors](joint-errors.png)",
        "",
        "Checkpoint selection never uses parameter errors. Raw training losses differ between "
        "kernels; compare matched errors, common unfaded diagonal loss and LSD. Actual update "
        "counts differ because the same patience rule may stop at different times. "
        "One trajectory per setting; results are exploratory, with no multi-seed robustness claim.",
        "",
    ]
    (DOCS / "README.md").write_text("\n".join(lines))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    for ax, mode in zip(axes, ("fresh", "plateau"), strict=True):
        for variant in VARIANTS:
            values = [lookup[c["target_id"], mode, variant]["joint_event_error"] for c in CASES]
            ax.plot(range(7), values, "o-", label=LABELS[variant])
        ax.set_yscale("log")
        ax.set_title(mode.title() + " start")
        ax.set_ylabel("Joint RMS matched error")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[-1].set_xticks(range(7), [c["target_id"] for c in CASES], rotation=25, ha="right")
    fig.savefig(DOCS / "joint-errors.png", dpi=160)
    plt.close(fig)
    print(json.dumps(paired, indent=2))


if __name__ == "__main__":
    main()
