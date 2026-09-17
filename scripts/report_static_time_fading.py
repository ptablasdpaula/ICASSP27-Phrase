"""Compare static quarter-second time-only fading with registered unfaded controls."""

import csv
import gzip
import json

import numpy as np
from test_static_time_fading import CASES, DOCS, ROOT, signature

METRICS = ("pitch_mae_cents", "onset_mae_ms", "joint_event_error", "log_spectral_distance_db")


def write_csv(name, rows):
    with (DOCS / name).open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    records, rows = [], []
    for index in range(14):
        with gzip.open(ROOT / f"{index:02d}.json.gz", "rt") as f:
            d = json.load(f)
        assert d["index"] == index and d["signature"] == signature()
        assert d["case"]["target_id"] == CASES[index // 2]["target_id"]
        assert d["variant"] == ("static_time" if index % 2 else "unfaded")
        assert d["frequency_horizon_hz"] is None and not d["fade_schedule"]
        assert d["time_horizon_seconds"] == (0.25 if index % 2 else None)
        fit = d["fit"]
        assert fit["updates"] <= 3000 and fit["best_loss"] == min(
            s["raw_loss"] for s in fit["trajectory"]
        )
        records.append(d)
        for state, key in (
            ("own_best", "metrics"),
            ("common_best", "common_metrics"),
            ("last", "last_metrics"),
        ):
            assert np.isfinite([d[key][m] for m in METRICS]).all()
            rows.append(
                dict(
                    target_id=d["case"]["target_id"],
                    variant=d["variant"],
                    state=state,
                    **{m: d[key][m] for m in METRICS},
                    updates=fit["updates"],
                    stopped_by=fit["stopped_by"],
                    recovered=d[key]["pitch_mae_cents"] < 1 and d[key]["onset_mae_ms"] < 1,
                )
            )
    lookup = {(r["target_id"], r["variant"], r["state"]): r for r in rows}
    assert len(lookup) == 42
    for i in range(0, 14, 2):
        for key in ("f0_hz", "onset_seconds"):
            assert (
                records[i]["fit"]["trajectory"][0][key]
                == records[i + 1]["fit"]["trajectory"][0][key]
            )
    summary = []
    for scope in ("all_seven", "other_five"):
        for state in ("own_best", "common_best", "last"):
            for variant in ("unfaded", "static_time"):
                group = [
                    r
                    for r in rows
                    if r["state"] == state
                    and r["variant"] == variant
                    and (scope == "all_seven" or r["target_id"] not in ("C04-T0005", "C08-T0000"))
                ]
                for metric in METRICS:
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
    write_csv("per_phrase.csv", rows)
    write_csv("summary.csv", summary)
    with gzip.open(DOCS / "raw-results.json.gz", "wt") as f:
        json.dump(records, f, separators=(",", ":"))
    lines = [
        "# Static time-only fading",
        "",
        "Complete: seven problem phrases × two fresh methods =14 fits.",
        "",
        "**Time horizon stays at 0.25 s from the start. "
        "Frequency accumulation is entirely unfaded.**",
        "",
        "[Protocol](protocol.md) · [Per-phrase metrics](per_phrase.csv) · "
        "[Statistics](summary.csv) · "
        "[Full trajectories](raw-results.json.gz)",
        "",
        "Four diagonals, fixed amplitudes, no Log-Weighing. Registered Adam and patience from "
        "update0, maximum3000 updates. Each method selects its own best fixed training loss.",
        "",
    ]
    for state, title in (
        ("own_best", "Best own training loss"),
        ("common_best", "Best common unfaded loss"),
        ("last", "Actual last iterate"),
    ):
        lines += [
            f"## {title}",
            "",
            "Pitch MAE (cents) / onset MAE (ms).",
            "",
            "| Case | Unfaded control | Static time-only .25s |",
            "|---|---:|---:|",
        ]
        for case in CASES:
            cid = case["target_id"]
            cells = []
            for v in ("unfaded", "static_time"):
                r = lookup[cid, v, state]
                cells.append(f"{r['pitch_mae_cents']:.3f} / {r['onset_mae_ms']:.3f}")
            lines.append("| " + cid + " | " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "Unlike the preceding progressive experiment, these runs enable patience immediately. "
        "That earlier experiment used2000 updates before patience. Compare this static method "
        "primarily with its newly rerun matched control, not directly with that earlier schedule. "
        "No target parameter error selects a checkpoint. "
        "One trajectory per setting on selected failures.",
        "",
    ]
    (DOCS / "README.md").write_text("\n".join(lines))
    print(
        json.dumps(
            [
                r
                for r in summary
                if r["scope"] == "all_seven"
                and r["variant"] == "static_time"
                and r["metric"] == "joint_event_error"
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
