"""Summarise the selected-case direction and pitch-swap interventions."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("docs/swap-escape-pilot")
LABELS = {
    "pitch_swap": "Greedy swaps (none accepted)",
    "cel_01": "Up, forward time",
    "cel_04": "Up, backward time",
    "cel_05": "Up, both time directions",
    "cel_10": "Down, both time directions",
    "cel_05_lw": "Up, both times + LW",
    "cel_15_lw": "All four + LW",
    "cel_03_lw": "Forward, both frequencies + LW",
}


def main():
    target = json.loads((ROOT / "provenance.json").read_text())["target"]
    baseline = json.loads(Path("docs/amplitude-pilot/baseline.json").read_text())
    data = {name: json.loads((ROOT / f"{name}.json").read_text()) for name in LABELS}
    rows = [
        "| Intervention | Updates | Canonical CeL at best | Pitch MAE (cents) | Onset MAE (ms) |",
        "|---|---:|---:|---:|---:|",
        f"| Unchanged baseline | — | {baseline['best_loss']:.8g} | 5.481 | 70.452 |",
    ]
    fig, axes = plt.subplots(
        2, 4, figsize=(14, 7), sharex=True, sharey=True, constrained_layout=True
    )
    for ax, (name, label) in zip(axes.flat, LABELS.items(), strict=True):
        d = data[name]
        m = d["metrics"]
        rows.append(
            f"| {label} | {d['updates']} | {d['canonical_cel_at_best']:.8g} | "
            f"{m['pitch_mae_cents']:.3f} | {m['onset_mae_ms']:.3f} |"
        )
        f, t = (np.array(d["best_phrase"][key]) for key in ["f0_hz", "onset_seconds"])
        ax.scatter(
            target["onset_seconds"],
            target["f0_hz"],
            marker="x",
            s=70,
            color="black",
            label="Target",
        )
        ax.scatter(t, f, s=70, facecolors="none", edgecolors="tab:blue", label="Fit")
        for i, j in enumerate(m["assignment"]):
            ax.plot(
                [t[i], target["onset_seconds"][j]],
                [f[i], target["f0_hz"][j]],
                color=".7",
                lw=0.8,
                zorder=0,
            )
        ax.set_title(label, fontsize=10)
        ax.set_yscale("log")
        ax.set_ylim(80, 320)
        ax.set_xlim(0.65, 1.8)
        ax.set_yticks([80, 120, 160, 240, 320], labels=["80", "120", "160", "240", "320"])
        ax.minorticks_off()
        ax.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.supylabel("Pitch (Hz)")
    fig.supxlabel("Onset (s); amplitudes fixed at 0.8")
    fig.savefig(ROOT / "fits.png", dpi=170)
    fig.savefig(ROOT / "fits.pdf")
    plt.close(fig)
    (ROOT / "table.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
