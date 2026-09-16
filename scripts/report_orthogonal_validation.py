"""Archive and compare six prespecified axis-only escape cases."""
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

RAW = Path("results/orthogonal-validation/raw")
OUT = Path("docs/orthogonal-validation")
VARIANTS = ["all_four", "frequency_only"]


def main():
    cases = json.loads(Path("docs/takeover-validation/cases.json").read_text())
    data = [json.loads((RAW / c["target_id"] / "summary.json").read_text()) for c in cases]
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    table = ["| Case | Baseline | Four orthogonals | Frequency only | Diagonal control |",
             "|---|---:|---:|---:|---:|"]
    costs = ["| Case | Four orthogonal updates | Frequency-only updates | Control updates |",
             "|---|---:|---:|---:|"]
    aggregate = {}
    methods = ["initial", *VARIANTS, "control"]
    for d in data:
        case = d["case"]["target_id"]
        row = [case]
        for method in methods:
            result = (d["initial"] if method == "initial" else d["control"]["final"]
                      if method == "control" else d["variants"][method]["final"])
            m = result["metrics"]
            updates = (0 if method == "initial" else d["control"]["updates"]
                       if method == "control" else d["variants"][method]["updates"])
            records.append(dict(case=case, events=d["case"]["events"], method=method,
                                canonical_loss=result["loss"],
                                pitch_mae_cents=m["pitch_mae_cents"],
                                onset_mae_ms=m["onset_mae_ms"],
                                joint_event_error=m["joint_event_error"],
                                relative_waveform_l2=m["relative_waveform_l2"], updates=updates))
            row.append(f"{m['pitch_mae_cents']:.3f} / {m['onset_mae_ms']:.3f}")
        table.append("| " + " | ".join(row) + " |")
        costs.append(f"| {case} | {d['variants']['all_four']['updates']} | "
                     f"{d['variants']['frequency_only']['updates']} | {d['control']['updates']} |")
    for method in VARIANTS:
        reductions = []
        strict = 0
        loss_gains = 0
        for d in data:
            final = d["variants"][method]["final"]
            control = d["control"]["final"]
            m = final["metrics"]
            reductions.append(100 * (1 - m["joint_event_error"]
                                     / control["metrics"]["joint_event_error"]))
            strict += int(m["pitch_mae_cents"] < 1 and m["onset_mae_ms"] < 1)
            loss_gains += int(final["loss"] < control["loss"] * .99)
        aggregate[method] = dict(joint_error_reduction_percent_vs_control=reductions,
                                 median_reduction_percent=statistics.median(reductions),
                                 gains_over_20_percent=sum(r > 20 for r in reductions),
                                 regressions_over_20_percent=sum(r < -20 for r in reductions),
                                 strict_recoveries=strict,
                                 canonical_loss_gains_over_1_percent=loss_gains)
    with (OUT / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (OUT / "table.md").write_text("Errors are pitch MAE (cents) / onset MAE (ms).\n\n"
                                   + "\n".join(table) + "\n")
    (OUT / "costs.md").write_text("\n".join(costs) + "\n")
    (OUT / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    for ax, metric, label in zip(axes, ["pitch_mae_cents", "onset_mae_ms"],
                                  ["Pitch MAE (cents)", "Onset MAE (ms)"], strict=True):
        for i, method in enumerate(methods):
            values = [r[metric] for r in records if r["method"] == method]
            ax.bar(np.arange(6) + (i - 1.5) * .19, np.maximum(values, 1e-6),
                   width=.18, label=method)
        ax.set_yscale("log")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(ncol=4, fontsize=8)
    axes[1].set_xticks(np.arange(6), [c["target_id"] for c in cases])
    fig.savefig(OUT / "comparison.png", dpi=170)
    fig.savefig(OUT / "comparison.pdf")
    plt.close(fig)
    with zipfile.ZipFile(OUT / "raw-results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(RAW.rglob("*.json")):
            archive.write(p, p.relative_to(RAW))
    print("\n".join(table))
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
