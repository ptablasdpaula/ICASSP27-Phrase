"""Inspect smooth rotation with fixed canonical reference and matched control."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("docs/swap-escape-pilot/rotation")


def main():
    data = {
        n: json.loads((ROOT / f"{n}.json").read_text()) for n in ["rotating", "uniform_control"]
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    tr = data["rotating"]["exploration_trajectory"]
    for q, label in enumerate(["right/up", "right/down", "left/up", "left/down"]):
        axes[0, 0].plot([s["update"] for s in tr], [s["weights"][q] for s in tr], label=label)
    axes[0, 0].set_ylabel("Directional weight")
    axes[0, 0].legend(fontsize=8)
    rows = [
        "| Run | Best exploratory canonical CeL | Polished CeL | "
        "Pitch MAE (cents) | Onset MAE (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, d in data.items():
        tr = d["exploration_trajectory"]
        steps = [s["update"] for s in tr]
        axes[0, 1].plot(steps, [s["best_canonical_loss"] for s in tr], label=name)
        axes[1, 0].plot(steps, [s["metrics"]["onset_mae_ms"] for s in tr], label=name)
        sampled = [s for s in tr if "uniform_gradient_norm" in s]
        axes[1, 1].plot(
            [s["update"] for s in sampled],
            [s["uniform_gradient_norm"] for s in sampled],
            label=name + " canonical",
        )
        if name == "rotating":
            axes[1, 1].plot(
                [s["update"] for s in sampled],
                [s["weighted_gradient_norm"] for s in sampled],
                label="rotating weighted",
            )
        p = d["polish"]
        m = p["metrics"]
        rows.append(
            f"| {name} | {d['best_exploration_loss']:.8g} | {p['best_loss']:.8g} | "
            f"{m['pitch_mae_cents']:.3f} | {m['onset_mae_ms']:.3f} |"
        )
    axes[0, 1].set_ylabel("Best original four-direction CeL")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].set_ylabel("Current matched onset MAE (ms)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].set_ylabel("Raw-coordinate gradient norm")
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("Exploration update")
        ax.grid(alpha=0.2)
    fig.savefig(ROOT / "rotation.png", dpi=170)
    fig.savefig(ROOT / "rotation.pdf")
    plt.close(fig)
    (ROOT / "table.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
