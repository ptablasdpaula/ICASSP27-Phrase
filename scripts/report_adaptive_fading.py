"""Report scheduled fading versus matched fresh unfaded controls."""

import csv
import gzip
import json

import numpy as np
import torch
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.targets import load_target
from test_adaptive_fading import CASES, DOCS, ROOT, Objective, schedule, signature


def csv_write(name, rows):
    with (DOCS / name).open("w") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    records, rows = [], []
    for index in range(14):
        with gzip.open(ROOT / f"{index:02d}.json.gz", "rt") as f:
            d = json.load(f)
        assert d["index"] == index and d["signature"] == signature()
        assert d["case"]["target_id"] == CASES[index // 2]["target_id"]
        assert d["variant"] == ("annealed" if index % 2 else "unfaded")
        fit = d["fit"]
        trace = fit["trajectory"]
        assert 2000 <= fit["updates"] <= 3000
        assert fit["best_loss"] == min(s["canonical_loss"] for s in trace)
        if d["variant"] == "annealed":
            assert trace[0]["horizons"] is None and trace[-1]["horizons"] == [1.0, 1.0]
        records.append(d)
        for state, key in (("best", "metrics"), ("last", "last_metrics")):
            rows.append(
                dict(
                    target_id=d["case"]["target_id"],
                    variant=d["variant"],
                    state=state,
                    **{
                        k: d[key][k]
                        for k in (
                            "pitch_mae_cents",
                            "onset_mae_ms",
                            "joint_event_error",
                            "log_spectral_distance_db",
                        )
                    },
                    updates=fit["updates"],
                    best_update=fit["best_update"],
                    stopped_by=fit["stopped_by"],
                    recovered=d[key]["pitch_mae_cents"] < 1 and d[key]["onset_mae_ms"] < 1,
                )
            )
    lookup = {(r["target_id"], r["variant"], r["state"]): r for r in rows}
    assert len(lookup) == 28
    for i in range(0, 14, 2):
        for key in ("f0_hz", "onset_seconds", "canonical_loss"):
            assert (
                records[i]["fit"]["trajectory"][0][key]
                == records[i + 1]["fit"]["trajectory"][0][key]
            )
    csv_write("per_phrase.csv", rows)
    summary = []
    for scope in ("all_seven", "other_five"):
        for state in ("best", "last"):
            for variant in ("unfaded", "annealed"):
                group = [
                    r
                    for r in rows
                    if r["state"] == state
                    and r["variant"] == variant
                    and (scope == "all_seven" or r["target_id"] not in ("C04-T0005", "C08-T0000"))
                ]
                for metric in (
                    "pitch_mae_cents",
                    "onset_mae_ms",
                    "joint_event_error",
                    "log_spectral_distance_db",
                ):
                    values = np.array([r[metric] for r in group])
                    delta = np.array(
                        [
                            r[metric] - lookup[r["target_id"], "unfaded", state][metric]
                            for r in group
                        ]
                    )
                    summary.append(
                        dict(
                            scope=scope,
                            state=state,
                            variant=variant,
                            metric=metric,
                            n=len(group),
                            mean=float(values.mean()),
                            std=float(values.std(ddof=1)),
                            median=float(np.median(values)),
                            better=int((delta < -1e-10).sum()),
                            worse=int((delta > 1e-10).sum()),
                            tied=int((np.abs(delta) <= 1e-10).sum()),
                            recovered=sum(r["recovered"] for r in group),
                        )
                    )
    csv_write("summary.csv", summary)
    with gzip.open(DOCS / "raw-results.json.gz", "wt") as f:
        json.dump(records, f, separators=(",", ":"))
    lines = [
        "# Gradual fading from a fresh start",
        "",
        "Complete: seven phrases × two methods = 14 fresh fits.",
        "",
        "[Protocol](protocol.md) · [Per-phrase metrics](per_phrase.csv) · "
        "[Statistics](summary.csv) · "
        "[Full trajectories](raw-results.json.gz)",
        "",
        "Fade horizons change continuously from infinity to 1 second / 1 octave over 2000 updates. "
        "Both methods share the schedule and enable patience after update 2000. "
        "No saved plateau is used. All amplitudes are fixed, and there is no Log-Weighing.",
        "",
        "## Best common-loss checkpoint",
        "",
        "Both methods select the strict-best original unfaded diagonal loss across the entire run. "
        "Entries are pitch MAE (cents) / onset MAE (ms).",
        "",
        "| Case | Unfaded control | Gradual fading |",
        "|---|---:|---:|",
    ]
    for case in CASES:
        cid = case["target_id"]
        cells = [
            f"{lookup[cid, v, 'best']['pitch_mae_cents']:.3f} / "
            f"{lookup[cid, v, 'best']['onset_mae_ms']:.3f}"
            for v in ("unfaded", "annealed")
        ]
        lines.append("| " + cid + " | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Actual last iterate",
        "",
        "No best-checkpoint selection in this table.",
        "",
        "| Case | Unfaded control | Gradual fading |",
        "|---|---:|---:|",
    ]
    for case in CASES:
        cid = case["target_id"]
        cells = [
            f"{lookup[cid, v, 'last']['pitch_mae_cents']:.3f} / "
            f"{lookup[cid, v, 'last']['onset_mae_ms']:.3f}"
            for v in ("unfaded", "annealed")
        ]
        lines.append("| " + cid + " | " + " | ".join(cells) + " |")
    lines += [
        "",
        "![Horizon schedule](schedule.png)",
        "",
        "![Best common loss](trajectories.png)",
        "",
        "One trajectory per setting on selected failures. The common-loss checkpoint may precede "
        "the final fade horizon. Final patience monitors each fixed terminal objective, while the "
        "reported checkpoint uses the same unfaded metric for both methods. The controls "
        "use this experiment’s schedule, not the earlier registered early-stopping schedule.",
        "",
    ]
    (DOCS / "README.md").write_text("\n".join(lines))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torch.set_num_threads(1)
    _, target = load_target(4, 6)
    obj = Objective(PhraseSynth().render(target).detach())
    steps = np.arange(1, 2001)
    h = np.array([obj.horizons(*schedule(int(s))) for s in steps])
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.semilogy(steps, h[:, 0], label="Time horizon (s)")
    ax.semilogy(steps, h[:, 1], label="Frequency horizon (octaves)")
    ax.set(xlabel="Adam update", ylabel="Fade-to-zero horizon", title="Update 0 is exactly unfaded")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(DOCS / "schedule.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(4, 2, figsize=(10, 10), constrained_layout=True)
    for ax, case in zip(axes.flat, CASES, strict=False):
        for d in records:
            if d["case"]["target_id"] == case["target_id"]:
                tr = d["fit"]["trajectory"]
                ax.semilogy(
                    [s["update"] for s in tr],
                    [s["best_canonical_loss"] for s in tr],
                    label=d["variant"],
                )
        ax.axvline(2000, color="gray", ls="--")
        ax.set_title(case["target_id"])
        ax.legend()
        ax.grid(alpha=0.2)
    axes.flat[-1].set_visible(False)
    fig.savefig(DOCS / "trajectories.png", dpi=160)
    plt.close(fig)
    print(
        json.dumps(
            [
                r
                for r in summary
                if r["scope"] == "all_seven"
                and r["variant"] == "annealed"
                and r["metric"] == "joint_event_error"
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
