"""Report actual versus trial improvements for cyclic assignment probes."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("docs/shadow-swap-pilot")


def main():
    runs = {
        name: json.loads((ROOT / f"{name}.json").read_text())
        for name in ["cyclic_one_step", "no_probe_control"]
    }
    rows = [
        "| Run | Accepted swaps | Best real CeL | "
        "Polished pitch MAE (cents) | Polished onset MAE (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    for name, d in runs.items():
        r = d["records"]
        p = d["polish"]
        m = p["metrics"]
        rows.append(
            f"| {name} | {sum(s['accepted'] for s in r)} | {d['best_loss']:.8g} | "
            f"{m['pitch_mae_cents']:.6f} | {m['onset_mae_ms']:.6f} |"
        )
        axes[0].plot([s["update"] for s in r], [s["best_loss"] for s in r], label=name)
    for pair in [[1, 2], [2, 3], [3, 4]]:
        r = [s for s in runs["cyclic_one_step"]["records"] if s["pair"] == pair]
        axes[1].plot([s["update"] for s in r], [s["swap_probability"] for s in r], label=str(pair))
    axes[0].set_ylabel("Best actual four-direction CeL")
    axes[0].set_yscale("log")
    axes[1].set_ylabel("Trial swap probability versus keep")
    axes[1].axhline(0.5, color="black", lw=0.8, linestyle="--")
    for ax in axes:
        ax.set_xlabel("Real update")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(ROOT / "cyclic.png", dpi=170)
    fig.savefig(ROOT / "cyclic.pdf")
    plt.close(fig)
    (ROOT / "table.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
    d = runs["cyclic_one_step"]
    accepted = [s for s in d["records"] if s["accepted"]]
    print("Accepted details", accepted)


if __name__ == "__main__":
    main()
