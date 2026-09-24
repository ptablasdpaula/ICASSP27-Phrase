#!/usr/bin/env python3
"""Render the 16-kHz single-pluck objective slices used in the paper."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib
import numpy as np
import torch
from icassp27_phrase.losses import PaperObjectives
from icassp27_phrase.runtime import configure_reproducibility, require_df2_backend
from icassp27_phrase.synth import PhraseSynth

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from icassp27_phrase.paths import OUTPUT as OUTPUT_ROOT
from icassp27_phrase.paths import REPO as ROOT
from icassp27_phrase.paths import resolve_path, scientific_signature

OUTPUT = OUTPUT_ROOT / "loss-slices"
PAPER_FIGURE = OUTPUT_ROOT / "figures/loss_sweeps.pdf"
NAMES = (
    "single_stft",
    "mss",
    "smooth_mss",
    "linear_jtfot",
    "cel",
    "sot_published_composite",
)
GROUPS = (NAMES[:3], NAMES[3:])
STYLE = {
    "single_stft": ("SS", "#171717", "--", 1.3),
    "mss": ("MSS", "#0072B2", "-", 1.3),
    "smooth_mss": ("SmoMSS", "#C62828", "-", 1.3),
    "linear_jtfot": (r"$\mathrm{TF}\mathcal{W}_2$", "#D55E00", "--", 1.3),
    "cel": (r"Ce$\mathcal{L}$", "#6A3D9A", "-", 1.3),
    "sot_published_composite": ("SOT", "#009E73", "-", 1.3),
}
LABEL_SIZE = 7.0
SHARED_LABEL_GAP_PT = 15.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute(device: str, batch_size: int) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    synth = PhraseSynth().to(device)
    axes = np.stack((np.linspace(-0.8, 0.8, 3201), np.linspace(-1.0, 1.0, 3201)))
    target_f0 = torch.tensor([[160.0]], dtype=torch.float64, device=device)
    target_onset = torch.tensor([[1.0]], dtype=torch.float64, device=device)
    with torch.no_grad():
        target = synth(target_f0, target_onset)[0]
        objectives = PaperObjectives(target, NAMES)
        self_losses = objectives.values(target)
        values = np.empty((2, len(NAMES), axes.shape[-1]), dtype=np.float64)
        for row, axis in enumerate(axes):
            for begin in range(0, len(axis), batch_size):
                displacement = torch.as_tensor(
                    axis[begin : begin + batch_size], dtype=torch.float64, device=device
                )[:, None]
                if row == 0:
                    f0 = torch.full_like(displacement, 160.0)
                    onset = 1.0 + displacement
                else:
                    f0 = 160.0 * 2.0**displacement
                    onset = torch.ones_like(displacement)
                audio = synth(f0.clamp(80.0, 320.0), onset.clamp(0.2, 1.8))
                terms = objectives.values(audio)
                for index, name in enumerate(NAMES):
                    values[row, index, begin : begin + len(displacement)] = (
                        terms[name].detach().cpu().numpy()
                    )
                if begin % (20 * batch_size) == 0:
                    print(f"row={row} {begin}/{len(axis)}", flush=True)
    if not np.isfinite(values).all() or np.any(values < -1e-12):
        raise FloatingPointError("loss slices contain invalid values")
    spans = np.ptp(values, axis=-1)
    if np.any(spans <= 0):
        raise ValueError("a loss slice has zero range")
    normalised = (values - values.min(axis=-1, keepdims=True)) / spans[..., None]
    target_relative = np.abs(values[..., 1600]) / spans
    # cuFFT evaluation leaves a negligible residual at the exact target.  Keep
    # the check well below plotting resolution while allowing that round-off.
    if target_relative.max() >= 1e-7:
        raise AssertionError(f"target residual is too large: {target_relative.max()}")
    validation = {
        "self_losses": {name: float(self_losses[name][0]) for name in NAMES},
        "target_loss_relative_to_sweep_range": target_relative.tolist(),
        "renderer": synth.provenance(),
    }
    return axes, values, {"normalised": normalised, "validation": validation}


def render(axes: np.ndarray, normalised: np.ndarray, destination: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": LABEL_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "path.simplify": False,
        }
    )
    # Restore the earlier wide treatment at its final print width. This keeps
    # labels at body-text size while reducing the figure's vertical footprint.
    figure, panels = plt.subplots(2, 2, figsize=(5.2, 3.45), sharey=True, squeeze=False)
    for row in range(2):
        for column, group in enumerate(GROUPS):
            axis = panels[row, column]
            axis.axvline(0.0, color="0.5", linestyle=":", linewidth=0.9, zorder=0)
            for name in group:
                label, colour, linestyle, width = STYLE[name]
                axis.plot(
                    axes[row],
                    normalised[row, NAMES.index(name)],
                    label=label,
                    color=colour,
                    linestyle=linestyle,
                    linewidth=width,
                    zorder=4 if linestyle == "--" else 2,
                )
            ticks = np.linspace(axes[row, 0], axes[row, -1], 5)
            axis.set(
                xlim=(axes[row, 0], axes[row, -1]),
                ylim=(-0.035, 1.07),
                xticks=ticks,
                yticks=(0.0, 0.5, 1.0),
            )
            axis.set_xticklabels(
                [f"{value:+.1f}" if value > 0 else f"{value:.1f}" for value in ticks]
            )
            axis.get_xticklabels()[0].set_horizontalalignment("left")
            axis.get_xticklabels()[-1].set_horizontalalignment("right")
            axis.tick_params(axis="x", labelsize=LABEL_SIZE)
            axis.tick_params(axis="y", labelleft=False, labelright=False, left=False)
            axis.grid(axis="y", alpha=0.18)
            for spine in axis.spines.values():
                spine.set_visible(True)
            axis.legend(
                loc="lower left",
                frameon=True,
                framealpha=0.9,
                facecolor="white",
                edgecolor="none",
                fontsize=LABEL_SIZE,
                handlelength=1.8,
                borderpad=0.25,
                labelspacing=0.25,
            )
    figure.subplots_adjust(
        left=0.085,
        right=0.997,
        top=0.997,
        bottom=0.18,
        hspace=0.50,
        wspace=0.035,
    )
    shared_y_label = figure.text(
        0.0,
        0.0,
        "Normalised Loss [0-1]",
        ha="center",
        va="center",
        rotation=90,
        fontsize=LABEL_SIZE,
    )
    shared_x_labels = []
    for label in ("Time shift (s)", "Frequency shift (octaves)"):
        shared_x_labels.append(
            figure.text(
                0.0,
                0.0,
                label,
                ha="center",
                va="center",
                fontsize=LABEL_SIZE,
            )
        )

    # Centre the shared labels on the panel block rather than on the canvas.
    # Work in display coordinates so the horizontal and vertical clearances
    # are the same physical distance and do not depend on the font size.
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    all_panel_boxes = [axis.get_window_extent(renderer) for axis in panels.flat]
    panel_left = min(box.x0 for box in all_panel_boxes)
    panel_bottom = min(box.y0 for box in all_panel_boxes)
    panel_top = max(box.y1 for box in all_panel_boxes)
    gap_pixels = SHARED_LABEL_GAP_PT * figure.dpi / 72.0

    def shift_text(text, dx_pixels: float, dy_pixels: float) -> None:
        x, y = text.get_position()
        text.set_position(
            (
                x + dx_pixels / figure.bbox.width,
                y + dy_pixels / figure.bbox.height,
            )
        )

    y_box = shared_y_label.get_window_extent(renderer)
    shift_text(
        shared_y_label,
        panel_left - gap_pixels - y_box.x1,
        (panel_bottom + panel_top) / 2.0 - (y_box.y0 + y_box.y1) / 2.0,
    )
    for row, text in enumerate(shared_x_labels):
        row_boxes = [panels[row, column].get_window_extent(renderer) for column in range(2)]
        row_left = min(box.x0 for box in row_boxes)
        row_right = max(box.x1 for box in row_boxes)
        row_bottom = min(box.y0 for box in row_boxes)
        text_box = text.get_window_extent(renderer)
        shift_text(
            text,
            (row_left + row_right) / 2.0 - (text_box.x0 + text_box.x1) / 2.0,
            row_bottom - gap_pixels - text_box.y1,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, metadata={"CreationDate": None, "ModDate": None})
    figure.savefig(destination.with_suffix(".png"), dpi=300)
    plt.close(figure)


def main() -> None:
    global OUTPUT, PAPER_FIGURE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=resolve_path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    OUTPUT = args.output / "loss-slices"
    PAPER_FIGURE = args.output / "figures/loss_sweeps.pdf"
    torch.set_num_threads(1)
    configure_reproducibility()
    require_df2_backend(args.device)
    signature, hashes = scientific_signature()
    axes, raw, result = compute(args.device, args.batch_size)
    normalised = result["normalised"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache = OUTPUT / "loss_sweeps.npz"
    np.savez_compressed(
        cache,
        displacement=axes,
        losses=raw,
        normalised=normalised,
        names=np.asarray(NAMES),
        signature=signature,
    )
    render(axes, normalised, PAPER_FIGURE)
    provenance = {
        "schema": "loss-sweeps-16k-v1",
        "signature": signature,
        "source_hashes": hashes,
        "source_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "dtype": "float64",
        "target": {"f0_hz": 160.0, "onset_seconds": 1.0},
        "points_per_slice": axes.shape[-1],
        "losses": list(NAMES),
        "normalisation": "independent min-max per objective and slice; no smoothing",
        "validation": result["validation"],
        "artifacts": {
            str(cache.relative_to(args.output)): sha256(cache),
            str(PAPER_FIGURE.relative_to(args.output)): sha256(PAPER_FIGURE),
            str(PAPER_FIGURE.with_suffix(".png").relative_to(args.output)): sha256(
                PAPER_FIGURE.with_suffix(".png")
            ),
        },
    }
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"status": "complete", "artifacts": provenance["artifacts"]}, indent=2))


if __name__ == "__main__":
    main()
