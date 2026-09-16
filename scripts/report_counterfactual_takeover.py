"""Plot full-state acceptance decisions and final recovery."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("docs/counterfactual-takeover")


def main():
    data = json.loads((ROOT / "result.json").read_text())
    target = json.loads(Path("docs/amplitude-pilot/provenance.json").read_text())["target"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True, constrained_layout=True)
    for ax, key, label in zip(
        axes,
        ["initial", "adopted", "final"],
        ["Stuck fit", "After paired trials", "After continuation"],
        strict=True,
    ):
        d = data[key]
        ax.scatter(
            target["onset_seconds"],
            target["f0_hz"],
            marker="x",
            s=85,
            color="black",
            label="Target",
        )
        ax.scatter(
            d["onset_seconds"],
            d["f0_hz"],
            s=95,
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
    rows = [
        "| Pair | Trial updates per branch | Best keep | Best swap | Decision |",
        "|---|---:|---:|---:|---|",
    ]
    for t in data["trials"]:
        decision = (
            "adopt swap"
            if t["accepted_swap"]
            else ("adopt keep" if t["chosen"] == 0 else "retain live")
        )
        rows.append(
            f"| {t['pair']} | {t['depth']} | {t['best_keep']:.8g} | "
            f"{t['best_swap']:.8g} | {decision} |"
        )
    (ROOT / "decisions.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
    print("FINAL", data["final"])


if __name__ == "__main__":
    main()
