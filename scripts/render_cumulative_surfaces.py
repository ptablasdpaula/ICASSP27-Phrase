#!/usr/bin/env python3
"""Show the landscape target and the four cumulative-power images in Section II."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from icassp27_phrase.losses import BidirectionalCumulativeEnergyDistance, reverse_cumsum
from icassp27_phrase.runtime import configure_reproducibility
from icassp27_phrase.synth import PhraseSynth
from matplotlib.patches import FancyArrowPatch
from render_loss_landscapes import (
    LANDSCAPE_FIGURE_STYLE,
    TARGET_F0_HZ,
    TARGET_ONSET_SECONDS,
    renderer_specs,
)

ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = (
    ("Up-right", False, False, (0.055, 0.055), (0.23, 0.23)),
    ("Down-right", False, True, (0.055, 0.945), (0.23, 0.77)),
    ("Up-left", True, False, (0.945, 0.055), (0.77, 0.23)),
    ("Down-left", True, True, (0.945, 0.945), (0.77, 0.77)),
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--output-stem", type=Path,
        default=ROOT / "paper/figures/cumulative_surfaces",
    )
    args = parser.parse_args()
    configure_reproducibility()
    torch.set_num_threads(1)
    configuration = renderer_specs()[0]
    synth = PhraseSynth(
        exciter_config=configuration.exciter,
        waveguide_config=configuration.waveguide,
    ).to(args.device)
    with torch.no_grad():
        pitches = torch.tensor([[TARGET_F0_HZ]], dtype=torch.float64, device=args.device)
        onsets = torch.tensor([[TARGET_ONSET_SECONDS]], dtype=torch.float64, device=args.device)
        target = synth(pitches, onsets)[0]
        loss = BidirectionalCumulativeEnergyDistance(target, sample_rate=synth.sample_rate)
        # Use the actual loss preprocessing, including its uncentred STFT.
        power = loss._power(target[None])[0]
        total_power = power.sum()
        surfaces = []
        validation = []
        for index, (name, time_reverse, frequency_reverse, _, _) in enumerate(DIRECTIONS):
            time_surface = (
                reverse_cumsum(power, 1) if time_reverse else power.cumsum(1)
            )
            surface = (
                reverse_cumsum(time_surface, 0)
                if frequency_reverse else time_surface.cumsum(0)
            )
            normalised = surface / total_power
            reference, scale = loss.references[index]
            feature = torch.sqrt(normalised.clamp_min(loss.sqrt_floor))
            torch.testing.assert_close(feature, reference, rtol=1e-12, atol=1e-14)
            torch.testing.assert_close(scale, total_power, rtol=1e-12, atol=1e-14)
            # Each image must be monotone along both of its stated directions.
            for axis, reverse in ((0, frequency_reverse), (1, time_reverse)):
                differences = torch.diff(normalised, dim=axis) * (-1 if reverse else 1)
                if float(differences.min()) < -1e-13:
                    raise AssertionError(f"{name} is not monotone along axis {axis}")
            surfaces.append(normalised.cpu().numpy())
            validation.append({
                "direction": name,
                "maximum_reference_feature_error": float((feature - reference).abs().max()),
                "surface_sha256": digest(normalised.cpu().numpy().tobytes()),
            })

    power_array = power.cpu().numpy()
    spectrogram_db = 10 * np.log10(np.maximum(power_array / power_array.max(), 1e-8))
    # STFT coordinates are window centres, rather than artificially shifted onsets.
    time_edges = (np.arange(power.shape[1] + 1) * loss.hop
                  + loss.n_fft / 2 - loss.hop / 2) / synth.sample_rate
    frequency_edges = (np.arange(power.shape[0] + 1) - 0.5) * synth.sample_rate / loss.n_fft
    frequency_edges[0] = 1e-3  # Positive plotting edge for DC on a log axis.
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.linewidth": 0.55,
        "pdf.fonttype": 42,
    })
    figure, axes = plt.subplots(1, 5, figsize=(7.2, 1.85), sharex=True, sharey=True)
    figure.subplots_adjust(left=0.075, right=0.91, bottom=0.21, top=0.88, wspace=0.06)
    common = dict(cmap="magma", vmin=-80, vmax=0, shading="flat", rasterized=True)
    spectral_image = axes[0].pcolormesh(time_edges, frequency_edges, spectrogram_db, **common)
    axes[0].set_title("Spectrogram")
    axes[0].set_ylabel("Frequency (Hz)")
    style = LANDSCAPE_FIGURE_STYLE
    for axis, surface, (_, time_reverse, frequency_reverse, start, _) in zip(
        axes[1:], surfaces, DIRECTIONS, strict=True
    ):
        displayed = 10 * np.log10(np.maximum(surface, 1e-8))
        axis.pcolormesh(time_edges, frequency_edges, displayed, **common)
        time_arrow = r"\leftarrow" if time_reverse else r"\rightarrow"
        frequency_arrow = r"\downarrow" if frequency_reverse else r"\uparrow"
        axis.set_title(r"$(d_t,d_f)= (" + time_arrow + "," + frequency_arrow + ")$", fontsize=7.5)
        # Four times the landscape arrow length and head size, keeping its fine shaft.
        direction = np.array([-1 if time_reverse else 1, -1 if frequency_reverse else 1])
        end = np.asarray(start) + direction * (4 * 2 * style.arrow_half_length / np.sqrt(2))
        axis.add_patch(FancyArrowPatch(
            start, end, transform=axis.transAxes, arrowstyle="-|>",
            mutation_scale=4 * style.arrow_mutation_scale,
            linewidth=style.arrow_linewidth, color=style.arrow_color,
            shrinkA=0, shrinkB=0, zorder=5,
        ))
    for axis in axes:
        axis.set_box_aspect(1)
        axis.set_yscale("log")
        axis.set_ylim(20, synth.sample_rate / 2)
        axis.set_xticks([0.5, 1, 1.5], ["0.5", "1", "1.5"])
        axis.set_xlim(time_edges[0], time_edges[-1])
        axis.set_yticks([20, 100, 1000, 2000], ["20", "100", "1000", "2000"])
        axis.minorticks_off()
        axis.set_xlabel("Time (s)", labelpad=1.5)
        axis.tick_params(length=2, width=0.5, pad=1.5)
    figure.canvas.draw()
    bounds = axes[-1].get_position()
    bar_axis = figure.add_axes([bounds.x1 + 0.014, bounds.y0, 0.012, bounds.height])
    colorbar = figure.colorbar(
        spectral_image, cax=bar_axis, orientation="vertical", ticks=[-80, -60, -40, -20, 0],
    )
    colorbar.ax.tick_params(length=1.5, width=0.5, pad=1, labelsize=6)
    colorbar.set_label("Power (dB)", fontsize=7, labelpad=3)
    colorbar.outline.set_linewidth(0.5)

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for suffix in ("pdf", "png"):
        output = args.output_stem.with_suffix(f".{suffix}")
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else None
        figure.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.025, metadata=metadata)
        artifacts[suffix] = digest(output.read_bytes())
    plt.close(figure)
    provenance = {
        "schema": "cumulative-surfaces-figure-v1",
        "target": {"f0_hz": TARGET_F0_HZ, "onset_seconds": TARGET_ONSET_SECONDS},
        "renderer": synth.provenance(),
        "device": args.device,
        "torch_version": torch.__version__,
        "stft": {"n_fft": loss.n_fft, "hop": loss.hop, "center": False,
                 "window": "periodic Hann", "shape": list(power.shape)},
        "display": {"spectrogram": "peak-relative power dB, [-80,0]",
                    "surfaces": "10*log10(S_q/m), shared [-80,0] dB display scale",
                    "frequency_axis": "logarithmic, 20-2000 Hz; sums retain all STFT bins",
                    "colorbar": "one vertical shared dB scale; distinct reference powers",
                    "arrows": "landscape style, 4x length/head size, original shaft width",
                    "direction_order": [item[0] for item in DIRECTIONS]},
        "target_audio_sha256": digest(target.cpu().numpy().tobytes()),
        "power_sha256": digest(power_array.tobytes()),
        "validation": validation,
        "artifacts": artifacts,
        "script_sha256": digest(Path(__file__).read_bytes()),
    }
    args.output_stem.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"output": str(args.output_stem), "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
