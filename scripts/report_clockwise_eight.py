"""Report paired clockwise-rotation plateau trials; development case kept separate."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import zipfile
from pathlib import Path

import matplotlib
import numpy as np
from test_clockwise_eight import EXPLORATION, POLISH, ROOT, VARIANTS

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("docs/clockwise-eight")


def main():
    cases = [dict(target_id="C04-T0005", events=4, target=6)] + json.loads(
        Path("docs/takeover-validation/cases.json").read_text()
    )
    data = {}
    signature = hashlib.sha256(Path("scripts/test_clockwise_eight.py").read_bytes()).hexdigest()
    for case in cases:
        for variant in VARIANTS:
            d = json.loads((ROOT / case["target_id"] / f"{variant}.json").read_text())
            assert d["script_sha256"] == signature
            assert d["updates"] == EXPLORATION + POLISH
            assert len(d["trajectory"]) == EXPLORATION + POLISH + 1
            values = [r["canonical_loss"] for r in d["trajectory"]]
            assert np.isfinite(values).all()
            assert abs(d["final"]["loss"] - min([d["baseline"]["loss"]] + values)) < 1e-12
            assert d["final"]["loss"] <= d["baseline"]["loss"] + 1e-12
            data[case["target_id"], variant] = d
    rows, table = (
        [],
        [
            "| Case | Start | Diagonal control | Uniform eight | Clockwise |",
            "|---|---:|---:|---:|---:|",
        ],
    )
    for case in cases:
        name = case["target_id"]
        cells = [name + (" (development)" if name == "C04-T0005" else "")]
        baseline = data[name, "clockwise"]["baseline"]
        for result in [baseline] + [
            data[name, v]["final"] for v in ("diagonal_control", "uniform_eight", "clockwise")
        ]:
            m = result["metrics"]
            cells.append(f"{m['pitch_mae_cents']:.3f} / {m['onset_mae_ms']:.3f}")
        table.append("| " + " | ".join(cells) + " |")
        for variant in VARIANTS:
            d = data[name, variant]
            assert abs(d["baseline"]["loss"] - baseline["loss"]) < 1e-12
            for stage in ("baseline", "exploration_endpoint", "last", "final"):
                result = d[stage]
                rows.append(
                    dict(
                        case=name,
                        variant=variant,
                        stage=stage,
                        loss=result["loss"],
                        **{
                            k: result["metrics"][k]
                            for k in (
                                "pitch_mae_cents",
                                "onset_mae_ms",
                                "joint_event_error",
                                "relative_waveform_l2",
                            )
                        },
                        best_update=d["best_update"],
                        updates=d["updates"],
                        wall_seconds=d["wall_seconds"],
                    )
                )
    aggregate = {}
    for variant in ("clockwise", "uniform_eight"):
        gains = []
        for case in cases[1:]:
            a = data[case["target_id"], variant]["final"]["metrics"]["joint_event_error"]
            b = data[case["target_id"], "diagonal_control"]["final"]["metrics"]["joint_event_error"]
            gains.append(100 * (1 - a / b))
        aggregate[variant] = dict(
            cases=[c["target_id"] for c in cases[1:]],
            joint_error_reduction_percent_vs_control=gains,
            median_reduction_percent=statistics.median(gains),
            improvements=sum(g > 1e-8 for g in gains),
            regressions=sum(g < -1e-8 for g in gains),
            gains_over_20_percent=sum(g > 20 for g in gains),
            regressions_over_20_percent=sum(g < -20 for g in gains),
            strict_recoveries=sum(
                data[c["target_id"], variant]["final"]["metrics"]["pitch_mae_cents"] < 1
                and data[c["target_id"], variant]["final"]["metrics"]["onset_mae_ms"] < 1
                for c in cases[1:]
            ),
        )
    with (OUT / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    (OUT / "table.md").write_text(
        "Pitch MAE (cents) / onset MAE (ms).\n\n" + "\n".join(table) + "\n"
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for i, variant in enumerate(("diagonal_control", "uniform_eight", "clockwise")):
        for ax, metric in zip(axes, ("pitch_mae_cents", "onset_mae_ms"), strict=True):
            values = [data[c["target_id"], variant]["final"]["metrics"][metric] for c in cases]
            ax.bar(
                np.arange(len(cases)) + (i - 1) * 0.25,
                np.maximum(values, 1e-6),
                width=0.24,
                label=variant.replace("_", " "),
            )
            ax.set_yscale("log")
            ax.set_xticks(range(len(cases)), [c["target_id"] for c in cases], rotation=35)
            ax.set_ylabel("Pitch MAE (cents)" if metric.startswith("pitch") else "Onset MAE (ms)")
            ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.savefig(OUT / "comparison.png", dpi=160)
    fig.savefig(OUT / "comparison.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), layout="constrained")
    for ax, case in zip(axes.flat, cases, strict=False):
        for variant in VARIANTS:
            d = data[case["target_id"], variant]
            (line,) = ax.plot(
                [r["update"] for r in d["trajectory"]],
                [r["canonical_loss"] for r in d["trajectory"]],
                alpha=0.25,
                linewidth=0.7,
            )
            ax.plot(
                [r["update"] for r in d["trajectory"]],
                [r["best_canonical_loss"] for r in d["trajectory"]],
                color=line.get_color(),
                label=variant,
                linewidth=1.4,
            )
        ax.axvline(EXPLORATION, color="black", linestyle="--", linewidth=0.7)
        ax.set_title(case["target_id"])
        ax.set_yscale("log")
        ax.set_xlabel("Updates")
        ax.set_ylabel("Canonical diagonal loss")
    axes.flat[-1].axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.savefig(OUT / "trajectories.png", dpi=130)
    plt.close(fig)
    with zipfile.ZipFile(OUT / "raw-results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for case in cases:
            for variant in VARIANTS:
                p = ROOT / case["target_id"] / f"{variant}.json"
                z.write(p, p.relative_to(ROOT))
    lines = [
        "# Clockwise eight-direction descent",
        "",
        "Seven plateau cases, three matched runs per case, 2,400 updates each. "
        "The first case is development; aggregate comparisons use only the six other failures.",
        "",
        "[Protocol](protocol.md) · [Full statistics](summary.csv) · "
        "[Raw trajectories](raw-results.zip)",
        "",
        "Clockwise ↗ → ↘ ↓ ↙ ← ↖ ↑ blends adjacent directional losses linearly, "
        "100 updates per sector. Two turns, then 800 diagonal-only refinement updates "
        "from the actual last state. All runs have the same phase lengths, LR schedule, "
        "initial loss scaling and fresh Adam at the phase boundary. All amplitudes stay fixed.",
        "",
        "Pitch MAE (cents) / onset MAE (ms):",
        "",
        *table,
        "",
        "![Final errors](comparison.png)",
        "",
        "## Six additional failed phrases",
        "",
    ]
    for variant, result in aggregate.items():
        lines.append(
            f"- {variant}: {result['improvements']}/6 improve joint event error versus "
            f"diagonal control; {result['regressions']}/6 worsen. Median relative "
            f"reduction {result['median_reduction_percent']:.2f}%. "
            f"{result['gains_over_20_percent']} gains and "
            f"{result['regressions_over_20_percent']} regressions exceed 20%. "
            f"{result['strict_recoveries']} cases reach <1 cent and <1 ms."
        )
    lines += [
        "",
        "![Canonical loss trajectories](trajectories.png)",
        "",
        "The dashed line marks the switch to diagonal refinement. These curves show the "
        "common canonical loss, not the rotating training objective: faint curves are current "
        "iterates, bold curves the retained minimum. Final reporting selects "
        "the best canonical-loss iterate across exploration/refinement and the starting "
        "incumbent. The CSV also contains the unselected endpoints. Parameter errors do "
        "not select checkpoints.",
        "",
        "Numerical sensitivity: the development fixed-eight run differs from the earlier "
        "eight-direction pilot despite identical starting coordinates and round-off-scale "
        "initial loss differences. The paths diverge over subsequent updates. "
        "[The audit](numerical-sensitivity.json) records this; it does not isolate the effect "
        "of equivalent summation order from CPU execution differences. Current paired variants "
        "share one process/node per case. Single-run gains are not a robustness result.",
        "",
        "One schedule, fixed initial phase, no Log-Weighing or directional scale matching. "
        "The clockwise schedule includes a 100-update warm-up; its effect is not separately "
        "ablated from the subsequent rotation. "
        "These selected plateau restarts do not establish effects on fresh-start training "
        "or general convergence. Equal backward counts do not mean equal runtime. A lower "
        "canonical loss can still mean larger matched parameter errors.",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines))
    print("\n".join(table))
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
