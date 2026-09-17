"""Render two review layouts from saved loss sweeps, without recomputation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/loss-sweeps"
STYLE = {
    "waveform_l1": (r"$L_1$", "#171717", "-", 0.45, 0.65),
    "waveform_mse": (r"$L_2$", "#49302B", "-", 0.45, 0.65),
    "linear_mss": ("Linear MSS", "#0072B2", "--", 1.3, 1.0),
    "smooth_mss": ("Smooth MSS", "#D55E00", "-.", 1.3, 1.0),
    "sot_published_composite": ("SOT", "#009E73", "--", 1.3, 1.0),
    "linear_jtfot": ("TFW", "#C18A00", "-.", 1.3, 1.0),
    "bidirectional_cumulative_energy": (r"Ce$\mathcal{L}$", "#6A3D9A", "-", 1.3, 1.0),
}
LAYOUTS = {
    "comparison-seven": (
        ("waveform_l1", "linear_mss", "smooth_mss"),
        (
            "waveform_mse",
            "sot_published_composite",
            "linear_jtfot",
            "bidirectional_cumulative_energy",
        ),
    ),
    "comparison-spectral": (
        ("linear_mss", "smooth_mss"),
        ("sot_published_composite", "linear_jtfot", "bidirectional_cumulative_energy"),
    ),
}


LAYOUTS["comparison-spectral-column"] = (
    (
        "linear_mss",
        "smooth_mss",
        "sot_published_composite",
        "linear_jtfot",
        "bidirectional_cumulative_energy",
    ),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=tuple(LAYOUTS))
    args = parser.parse_args()
    source = OUTPUT / "raw-sweeps.npz"
    original_hash = sha(source)
    previous = json.loads((OUTPUT / "provenance.json").read_text())
    assert previous["artifact_sha256"][source.name] == original_hash
    with np.load(source) as saved:
        axes = saved["displacement"]
        raw = saved["losses"]
        normalised = saved["normalised"]
        names = saved["names"].tolist()
    assert axes.shape == (2, 3201) and raw.shape == (2, 7, 3201)
    assert np.isfinite(raw).all() and np.all(axes[:, 1600] == 0)
    np.testing.assert_array_equal(
        normalised,
        (raw - raw.min(axis=-1, keepdims=True)) / np.ptp(raw, axis=-1)[..., None],
    )
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "path.simplify": False,
        }
    )
    outputs = {
        p.name: sha(p)
        for stem in LAYOUTS
        for ext in ("png", "pdf")
        if (p := OUTPUT / f"{stem}.{ext}").exists()
    }
    for stem, groups in LAYOUTS.items():
        if args.layout and stem != args.layout:
            continue
        columns = len(groups)
        fig, panels = plt.subplots(
            2, columns, figsize=(5.5 if columns == 1 else 10, 5.9), sharey=True, squeeze=False
        )
        for row in range(2):
            for col, group in enumerate(groups):
                ax = panels[row, col]
                ax.axvline(0, color="0.5", linestyle=":", linewidth=0.9, zorder=0)
                for name in group:
                    label, colour, style, width, alpha = STYLE[name]
                    ax.plot(
                        axes[row],
                        normalised[row, names.index(name)],
                        label=label,
                        color=colour,
                        linestyle=style,
                        linewidth=width,
                        alpha=alpha,
                        zorder=1 if name.startswith("waveform") else 3,
                    )
                ax.set(
                    xlim=(axes[row, 0], axes[row, -1]),
                    ylim=(-0.035, 1.07),
                    xticks=np.linspace(axes[row, 0], axes[row, -1], 5),
                    yticks=[0, 0.5, 1],
                    xlabel="Onset displacement (s)" if row == 0 else "Pitch displacement (octaves)",
                )
                if col == 0:
                    ax.set_ylabel("Normalised loss")
                ax.grid(axis="y", alpha=0.18)
                if row == 0:
                    ax.legend(
                        loc="lower center",
                        bbox_to_anchor=(0.5, 1.07),
                        ncol=3 if len(group) == 5 else 2 if len(group) == 4 else len(group),
                        frameon=False,
                        fontsize=9,
                        handlelength=2.5,
                        columnspacing=1.1,
                    )
        fig.subplots_adjust(
            left=0.15 if columns == 1 else 0.075,
            right=0.955,
            bottom=0.105,
            top=0.84,
            hspace=0.40,
            wspace=0.19,
        )
        for ext in ("png", "pdf"):
            path = OUTPUT / f"{stem}.{ext}"
            fig.savefig(path, dpi=180)
            outputs[path.name] = sha(path)
            print(path, flush=True)
        plt.close(fig)
    assert sha(source) == original_hash
    provenance = {
        "source": "raw-sweeps.npz",
        "source_sha256": original_hash,
        "script_sha256": sha(Path(__file__)),
        "layouts": LAYOUTS,
        "styles": STYLE,
        "normalisation": "Unchanged saved independent min-max per loss and sweep",
        "points_per_curve": 3201,
        "smoothing": False,
        "downsampling": False,
        "manuscript_changed": False,
        "artifact_sha256": outputs,
        "checks": {
            "source_hash_matches": True,
            "normalisation_matches_exactly": True,
            "raw_arrays_unchanged": True,
            "all_values_finite": True,
        },
    }
    (OUTPUT / "layouts.provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
