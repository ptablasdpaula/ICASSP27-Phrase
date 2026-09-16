"""Compare diagonal refinement from last versus selected orthogonal iterates."""
from __future__ import annotations

import csv
import json
import statistics
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("results/last-orthogonal-refinement/raw")
OUT = Path("docs/last-orthogonal-refinement")
VARIANTS = ["all_four", "frequency_only"]


def main():
    cases = json.loads(Path("docs/takeover-validation/cases.json").read_text())
    data = {(c["target_id"], v): json.loads((ROOT / c["target_id"] / f"{v}.json").read_text())
            for c in cases for v in VARIANTS}
    rows = []
    table = ["| Case | Variant | Previous refinement | Last orthogonal state | New refinement |",
             "|---|---|---:|---:|---:|"]
    costs = ["| Case | Variant | Previous refinement updates | New refinement updates | Stop |",
             "|---|---|---:|---:|---|"]
    aggregate = {}
    for (case, variant), d in data.items():
        cells = [case, variant]
        for stage, result in [("previous", d["previous"]["final"]),
                              ("last_orthogonal", d["start"]), ("new", d["final"])]:
            m = result["metrics"]
            rows.append(dict(case=case, variant=variant, stage=stage, loss=result["loss"],
                             pitch_mae_cents=m["pitch_mae_cents"],
                             onset_mae_ms=m["onset_mae_ms"],
                             joint_event_error=m["joint_event_error"],
                             relative_waveform_l2=m["relative_waveform_l2"]))
            cells.append(" / ".join(f"{m[k]:.3g}" if 0 < m[k] < .001 else f"{m[k]:.3f}"
                                    for k in ["pitch_mae_cents", "onset_mae_ms"]))
        table.append("| " + " | ".join(cells) + " |")
        costs.append(f"| {case} | {variant} | {d['previous']['updates'] - 1600} | "
                     f"{d['fit']['updates']} | {d['fit']['stopped_by']} |")
    for variant in VARIANTS:
        subset = [d for (c, v), d in data.items() if v == variant]
        reductions = [100 * (1 - d["final"]["metrics"]["joint_event_error"] /
                            d["previous"]["final"]["metrics"]["joint_event_error"]) for d in subset]
        aggregate[variant] = dict(
            joint_error_reduction_percent_vs_previous=reductions,
            median_joint_error_reduction_percent=statistics.median(reductions),
            gains_over_20_percent=sum(r > 20 for r in reductions),
            regressions_over_20_percent=sum(r < -20 for r in reductions),
            strict_recoveries=sum(d["final"]["metrics"]["pitch_mae_cents"] < 1 and
                                  d["final"]["metrics"]["onset_mae_ms"] < 1 for d in subset),
            canonical_loss_improvements_over_1_percent=sum(
                d["final"]["loss"] < d["previous"]["final"]["loss"] * .99 for d in subset))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "table.md").write_text("Pitch MAE (cents) / onset MAE (ms).\n\n"
                                   + "\n".join(table) + "\n")
    (OUT / "costs.md").write_text("\n".join(costs) + "\n")
    (OUT / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, constrained_layout=True)
    for col, variant in enumerate(VARIANTS):
        axes[0, col].set_title("Four orthogonals" if col == 0 else "Frequency only")
        for i, stage in enumerate(["previous", "last_orthogonal", "new"]):
            subset = [r for r in rows if r["variant"] == variant and r["stage"] == stage]
            for row, metric in enumerate(["pitch_mae_cents", "onset_mae_ms"]):
                ax = axes[row, col]
                ax.bar(np.arange(6) + (i - 1) * .25,
                       np.maximum([r[metric] for r in subset], 1e-6), width=.24, label=stage)
                ax.set_yscale("log")
                ax.grid(axis="y", alpha=.2)
        axes[1, col].set_xticks(np.arange(6), [c["target_id"] for c in cases], rotation=30)
    axes[0, 0].set_ylabel("Pitch MAE (cents)")
    axes[1, 0].set_ylabel("Onset MAE (ms)")
    axes[0, 0].legend(fontsize=8)
    fig.savefig(OUT / "comparison.png", dpi=170)
    fig.savefig(OUT / "comparison.pdf")
    plt.close(fig)
    with zipfile.ZipFile(OUT / "raw-results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ROOT.rglob("*.json")):
            z.write(p, p.relative_to(ROOT))
    print("\n".join(table))
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
