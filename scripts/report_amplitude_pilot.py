"""Plot the saved paired amplitude pilot without importing Torch."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("docs/amplitude-pilot")
NAMES = ["baseline", "restart_fixed", "restart_amplitude", "initial_amplitude"]
LABELS = [
    "Fixed: original start",
    "Fixed: restart",
    "Free amplitude: restart",
    "Free amplitude: original start",
]


def main():
    provenance = json.loads((ROOT / "provenance.json").read_text())
    target = provenance["target"]
    data = [json.loads((ROOT / f"{name}.json").read_text()) for name in NAMES]
    fig, axes = plt.subplots(
        1, 4, figsize=(14, 3.6), sharex=True, sharey=True, constrained_layout=True
    )
    for ax, label, result in zip(axes, LABELS, data, strict=True):
        f, t = (np.array(result["best_phrase"][key]) for key in ["f0_hz", "onset_seconds"])
        ax.scatter(
            target["onset_seconds"],
            target["f0_hz"],
            marker="x",
            s=80,
            color="black",
            label="Target (a = 0.8)",
        )
        ax.scatter(
            t,
            f,
            s=80 * np.array(result["best_amplitudes"]) / 0.8,
            facecolors="none",
            edgecolors="tab:blue",
            label="Fit (area ∝ amplitude)",
        )
        for i, j in enumerate(result["metrics"]["assignment"]):
            ax.plot(
                [t[i], target["onset_seconds"][j]],
                [f[i], target["f0_hz"][j]],
                color=".7",
                lw=0.8,
                zorder=0,
            )
            ax.annotate(
                f"{result['best_amplitudes'][i]:.2f}",
                (t[i], f[i]),
                xytext=(3, 7),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_title(label, fontsize=10)
        ax.set_yscale("log")
        ax.set_ylim(80, 320)
        ax.set_xlim(0.65, 1.8)
        ax.set_yticks([80, 120, 160, 240, 320], labels=["80", "120", "160", "240", "320"])
        ax.minorticks_off()
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Pitch (Hz)")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.supxlabel("Onset (s); annotations are fitted excitation amplitudes")
    fig.savefig(ROOT / "fits.png", dpi=180)
    fig.savefig(ROOT / "fits.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for label, result in zip(LABELS, data, strict=True):
        tr = result["trajectory"]
        u = [s["update"] for s in tr]
        axes[0, 0].plot(u, [s["best_loss"] for s in tr], label=label)
        axes[0, 1].plot(u, [s["metrics"]["pitch_mae_cents"] for s in tr], label=label)
        axes[1, 0].plot(u, [s["metrics"]["onset_mae_ms"] for s in tr], label=label)
    for index, name in [(2, "Restart"), (3, "Original start")]:
        tr = data[index]["trajectory"]
        for event in range(4):
            axes[1, 1].plot(
                [s["update"] for s in tr],
                [s["amplitudes"][event] for s in tr],
                linestyle="-" if index == 2 else "--",
                color=f"C{event}",
                label=f"{name}, event {event + 1}",
            )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("Best CeL objective")
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].set_ylabel("Current pitch MAE (cents)")
    axes[1, 0].set_ylabel("Current onset MAE (ms)")
    axes[1, 1].set_ylabel("Current excitation amplitude")
    axes[1, 1].axhline(0.8, color="black", lw=0.8, alpha=0.5)
    axes[1, 1].legend(fontsize=7, ncol=2)
    for ax in axes.flat:
        ax.set_xlabel("Adam updates in this fit")
        ax.grid(alpha=0.2)
    fig.savefig(ROOT / "trajectories.png", dpi=160)
    fig.savefig(ROOT / "trajectories.pdf")
    plt.close(fig)
    diagnostics = {}
    for name, result in zip(NAMES, data, strict=True):
        phrase = result["best_phrase"]
        order = np.argsort(phrase["onset_seconds"])
        ordered_f = np.array(phrase["f0_hz"])[order]
        diagnostics[name] = {
            "time_ordered_pitch_mae_cents": float(
                np.abs(1200 * np.log2(ordered_f / np.array(target["f0_hz"]))).mean()
            ),
            "amplitude_min_over_trajectory": float(
                min(min(s["amplitudes"]) for s in result["trajectory"])
            ),
            "amplitude_max_over_trajectory": float(
                max(max(s["amplitudes"]) for s in result["trajectory"])
            ),
            "stopped_by": result["stopped_by"],
        }
    (ROOT / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    rows = [
        "| Condition | Updates | Best CeL | Pitch MAE (cents) | Onset MAE (ms) | Amplitudes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for label, result in zip(LABELS, data, strict=True):
        m = result["metrics"]
        amps = ", ".join(f"{a:.3f}" for a in result["best_amplitudes"])
        rows.append(
            f"| {label} | {result['updates']} | {result['best_loss']:.8g} | "
            f"{m['pitch_mae_cents']:.3f} | {m['onset_mae_ms']:.3f} | {amps} |"
        )
    (ROOT / "table.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
