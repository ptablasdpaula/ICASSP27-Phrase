"""Compare independent and cumulative onset parameterisations on the saved case."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("docs/ordered-time-pilot")
PRIOR = Path("docs/amplitude-pilot")


def main():
    paths = [
        PRIOR / "baseline.json",
        PRIOR / "restart_fixed.json",
        ROOT / "ordered_restart.json",
        ROOT / "ordered_initial.json",
    ]
    labels = [
        "Independent: original start",
        "Independent: restart",
        "Cumulative: restart",
        "Cumulative: original start",
    ]
    data = [json.loads(p.read_text()) for p in paths]
    target = json.loads((ROOT / "provenance.json").read_text())["target"]
    fig, axes = plt.subplots(
        1, 4, figsize=(14, 3.6), sharex=True, sharey=True, constrained_layout=True
    )
    for ax, label, result in zip(axes, labels, data, strict=True):
        f, t = (np.array(result["best_phrase"][key]) for key in ["f0_hz", "onset_seconds"])
        ax.scatter(
            target["onset_seconds"],
            target["f0_hz"],
            marker="x",
            s=80,
            color="black",
            label="Target",
        )
        ax.scatter(t, f, s=80, facecolors="none", edgecolors="tab:blue", label="Fit")
        for i, j in enumerate(result["metrics"]["assignment"]):
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
        ax.set_xlim(0.2, 1.8)
        ax.set_yticks([80, 120, 160, 240, 320], labels=["80", "120", "160", "240", "320"])
        ax.minorticks_off()
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Pitch (Hz)")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.supxlabel("Onset (s); all excitation amplitudes fixed at 0.8")
    fig.savefig(ROOT / "fits.png", dpi=180)
    fig.savefig(ROOT / "fits.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for label, result in zip(labels, data, strict=True):
        tr = result["trajectory"]
        u = [s["update"] for s in tr]
        axes[0].plot(u, [s["best_loss"] for s in tr], label=label)
        axes[1].plot(u, [s["metrics"]["pitch_mae_cents"] for s in tr])
        axes[2].plot(u, [s["metrics"]["onset_mae_ms"] for s in tr])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Best CeL objective")
    axes[0].legend(fontsize=7)
    axes[1].set_ylabel("Current matched pitch MAE (cents)")
    axes[2].set_ylabel("Current matched onset MAE (ms)")
    for ax in axes:
        ax.set_xlabel("Adam updates in this fit")
        ax.grid(alpha=0.2)
    fig.savefig(ROOT / "trajectories.png", dpi=180)
    fig.savefig(ROOT / "trajectories.pdf")
    plt.close(fig)
    rows = [
        "| Condition | Updates | Best CeL | Pitch MAE (cents) | Onset MAE (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    diagnostics = {}
    for label, result in zip(labels, data, strict=True):
        m = result["metrics"]
        rows.append(
            f"| {label} | {result['updates']} | {result['best_loss']:.8g} | "
            f"{m['pitch_mae_cents']:.3f} | {m['onset_mae_ms']:.3f} |"
        )
        p = result["best_phrase"]
        order = np.argsort(p["onset_seconds"])
        diagnostics[label] = dict(
            time_ordered_pitch_mae_cents=float(
                np.abs(
                    1200 * np.log2(np.array(p["f0_hz"])[order] / np.array(target["f0_hz"]))
                ).mean()
            ),
            min_gap_seconds=float(np.diff(np.sort(p["onset_seconds"])).min()),
            stopped_by=result["stopped_by"],
        )
    (ROOT / "table.md").write_text("\n".join(rows) + "\n")
    (ROOT / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
