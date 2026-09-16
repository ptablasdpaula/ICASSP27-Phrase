"""Summarise the selected-case axis-only escape experiment."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("docs/orthogonal-escape-pilot")
NAMES = ["all_four", "frequency_only", "diagonal_control"]


def main():
    rows = ["| Restart | Final diagonal CeL | Pitch MAE (cents) | "
            "Onset MAE (ms) | Polish updates |",
            "|---|---:|---:|---:|---:|"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for name in NAMES:
        data = json.loads((ROOT / f"{name}.json").read_text())
        p = data["polish"]
        m = p["metrics"]
        rows.append(f"| {name} | {p['best_loss']:.9g} | {m['pitch_mae_cents']:.6f} | "
                    f"{m['onset_mae_ms']:.6f} | {p['updates']} |")
        tr = data["exploration_trajectory"]
        steps = [r["update"] for r in tr]
        for ax, values in zip(axes, [[r["canonical_loss"] for r in tr],
                                    [r["metrics"]["pitch_mae_cents"] for r in tr],
                                    [r["metrics"]["onset_mae_ms"] for r in tr]], strict=True):
            ax.plot(steps, values, label=name, linewidth=0.8)
    for ax, label in zip(axes, ["Original diagonal CeL", "Pitch MAE (cents)", "Onset MAE (ms)"],
                         strict=True):
        ax.set_xlabel("Exploration update")
        ax.set_ylabel(label)
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.savefig(ROOT / "comparison.png", dpi=170)
    fig.savefig(ROOT / "comparison.pdf")
    plt.close(fig)
    (ROOT / "table.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
