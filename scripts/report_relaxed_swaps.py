"""Select the best relaxed pitch swap by original CeL and plot its fit."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("docs/swap-escape-pilot/relaxed-swaps")


def main():
    paths = sorted(ROOT.glob("swap_*.json"))
    assert len(paths) == 6
    data = [json.loads(p.read_text()) for p in paths]
    baseline = json.loads(Path("docs/amplitude-pilot/baseline.json").read_text())
    winner = min(data + [baseline], key=lambda d: d["best_loss"])
    selection = dict(
        pair=winner.get("pair"), best_loss=winner["best_loss"], metrics=winner["metrics"]
    )
    (ROOT / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    rows = [
        "| Swapped slots | Updates | Best canonical CeL | Pitch MAE (cents) | Onset MAE (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for d in data:
        m = d["metrics"]
        rows.append(
            f"| {d['pair']} | {d['updates']} | {d['best_loss']:.8g} | "
            f"{m['pitch_mae_cents']:.6f} | {m['onset_mae_ms']:.6f} |"
        )
    (ROOT / "table.md").write_text("\n".join(rows) + "\n")
    target = json.loads(Path("docs/swap-escape-pilot/provenance.json").read_text())["target"]
    fig, axes = plt.subplots(
        1, 2, figsize=(9, 3.6), sharex=True, sharey=True, constrained_layout=True
    )
    for ax, d, label in zip(
        axes, [baseline, winner], ["Stuck baseline", "Best relaxed swap"], strict=True
    ):
        p = d["best_phrase"]
        ax.scatter(
            target["onset_seconds"],
            target["f0_hz"],
            marker="x",
            s=90,
            color="black",
            label="Target",
        )
        ax.scatter(
            p["onset_seconds"],
            p["f0_hz"],
            s=100,
            facecolors="none",
            edgecolors="tab:blue",
            label="Fit",
        )
        ax.set_title(label)
        ax.set_yscale("log")
        ax.set_yticks([120, 160, 240, 320], labels=["120", "160", "240", "320"])
        ax.minorticks_off()
        ax.grid(alpha=0.15)
        ax.set_xlabel("Onset (s)")
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel("Pitch (Hz)")
    fig.savefig(ROOT / "recovery.png", dpi=180)
    fig.savefig(ROOT / "recovery.pdf")
    plt.close(fig)
    print("\n".join(rows))


if __name__ == "__main__":
    main()
