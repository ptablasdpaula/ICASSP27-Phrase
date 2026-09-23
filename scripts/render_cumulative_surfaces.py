#!/usr/bin/env python3
"""Show a sine sweep or waveguide phrase and its four cumulative-power images."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cels import CumulativeEnergyLoss, Direction, STFTPower
from icassp27_phrase.config import SAMPLE_RATE
from icassp27_phrase.losses import CEL_HOP, CEL_N_FFT
from icassp27_phrase.runtime import configure_reproducibility
from matplotlib.patches import FancyArrowPatch
from render_loss_landscapes import LANDSCAPE_FIGURE_STYLE

ROOT = Path(__file__).resolve().parents[1]
DURATION_SECONDS = 2.0
START_HZ = 20.0
END_HZ = 1000.0
DIRECTIONS = (
    (Direction.RIGHT_UP, False, False, (0.055, 0.055), (0.23, 0.23)),
    (Direction.RIGHT_DOWN, False, True, (0.055, 0.945), (0.23, 0.77)),
    (Direction.LEFT_UP, True, False, (0.945, 0.055), (0.77, 0.23)),
    (Direction.LEFT_DOWN, True, True, (0.945, 0.945), (0.77, 0.77)),
)


def reverse_cumsum(value: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.flip(torch.cumsum(torch.flip(value, (dim,)), dim), (dim,))


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-frequency", type=float, default=END_HZ,
                        help="Upper display frequency in Hz; accumulation uses all bins.")
    parser.add_argument("--decibels", action="store_true",
                        help="Display normalised power in dB without changing accumulation.")
    parser.add_argument("--db-floor", type=float, default=-40.0)
    parser.add_argument("--descending-from-zero", action="store_true",
                        help="Illustration only: descending staircase at times 2*i/N.")
    parser.add_argument("--staircase", type=int, choices=(4, 6, 8),
                        help="80-to-320 Hz log-spaced notes with equal inter-note and edge gaps.")
    parser.add_argument(
        "--event", nargs=2, type=float, action="append", metavar=("HZ", "SECONDS"),
        help="Waveguide event; repeat for a phrase. Omit to render the sine sweep.",
    )
    parser.add_argument(
        "--output-stem", type=Path,
        default=ROOT / "paper/figures/cumulative_surfaces",
    )
    args = parser.parse_args()
    if not np.isfinite(args.db_floor) or args.db_floor >= 0:
        parser.error("--db-floor must be finite and negative")
    if args.descending_from_zero and not args.staircase:
        parser.error("--descending-from-zero requires --staircase")
    if args.staircase:
        if args.event:
            parser.error("use either --staircase or --event")
        args.event = list(zip(np.geomspace(80, 320, args.staircase).tolist(),
                             (np.arange(1, args.staircase + 1) * DURATION_SECONDS / (args.staircase + 1)).tolist()))
        if args.descending_from_zero:
            args.event = list(zip(
                np.geomspace(320, 80, args.staircase).tolist(),
                (np.arange(args.staircase) * DURATION_SECONDS / args.staircase).tolist(),
            ))
    if not START_HZ < args.max_frequency <= SAMPLE_RATE / 2:
        parser.error("--max-frequency must exceed 20 Hz and not exceed Nyquist")
    configure_reproducibility()
    torch.set_num_threads(1)
    with torch.no_grad():
        times = torch.arange(int(SAMPLE_RATE * DURATION_SECONDS),
                             dtype=torch.float64, device=args.device) / SAMPLE_RATE
        rate = np.log(END_HZ / START_HZ) / DURATION_SECONDS
        # Integrate instantaneous frequency before taking the sine.
        phase = (2 * np.pi * START_HZ / rate) * torch.expm1(rate * times)
        target = torch.sin(phase)
        target_description = {
            "type": "constant-amplitude logarithmic sinusoidal sweep",
            "start_hz": START_HZ, "end_hz": END_HZ,
            "duration_seconds": DURATION_SECONDS, "sample_rate": SAMPLE_RATE,
            "amplitude": 1.0, "initial_phase_radians": 0.0,
            "instantaneous_frequency": "f(t)=20*50**(t/2)",
            "phase": "2*pi*20*(exp(a*t)-1)/a, a=log(50)/2",
        }
        if args.event:
            from icassp27_phrase.synth import PhraseSynth

            synth = PhraseSynth().to(args.device)
            events = torch.tensor(args.event, dtype=torch.float64, device=args.device)
            # Relax the onset-domain check only within this illustration's render.
            # The experiment configuration and synthesis algorithm are unchanged.
            onset_domain = (
                patch("icassp27_phrase.waveguide.ONSET_BOUNDS_SECONDS", (0., DURATION_SECONDS))
                if args.descending_from_zero else nullcontext()
            )
            with onset_domain:
                target = synth(events[:, 0][None], events[:, 1][None])[0]
            target_description = {
                "type": "waveguide phrase",
                "events": [{"f0_hz": hz, "onset_seconds": onset}
                           for hz, onset in args.event],
                "duration_seconds": DURATION_SECONDS, "sample_rate": SAMPLE_RATE,
                "synthesiser": synth.provenance(),
                "staircase_events": args.staircase,
                "staircase_timing": "2*i/N, i=0,...,N-1" if args.descending_from_zero else "equal edge gaps",
            }
        transform = STFTPower(
            sample_rate=SAMPLE_RATE,
            n_fft=CEL_N_FFT,
            hop_length=CEL_HOP,
            center=False,
        ).to(device=args.device, dtype=torch.float64)
        power = transform(target[None])[0]
        frequency, frame_time = transform.coordinates(power)
        loss = CumulativeEnergyLoss(reduction="none")
        bound = loss.bind(power, frequency=frequency, time=frame_time)
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
            reference = bound.reference[index]
            feature = torch.sqrt(normalised.clamp_min(loss.eps))
            torch.testing.assert_close(feature, reference, rtol=1e-12, atol=1e-14)
            torch.testing.assert_close(bound.mass.squeeze(), total_power, rtol=1e-12, atol=1e-14)
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
    spectrogram_normalised = power_array / power_array.max()
    def display(value):
        return 10 * np.log10(np.maximum(value, 10 ** (args.db_floor / 10))) if args.decibels else value
    # STFT coordinates are window centres, rather than artificially shifted onsets.
    time_edges = (np.arange(power.shape[1] + 1) * CEL_HOP
                  + CEL_N_FFT / 2 - CEL_HOP / 2) / SAMPLE_RATE
    frequency_edges = (np.arange(power.shape[0] + 1) - 0.5) * SAMPLE_RATE / CEL_N_FFT
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
    common = dict(cmap="magma", vmin=args.db_floor if args.decibels else 0,
                  vmax=0 if args.decibels else 1, shading="flat", rasterized=True)
    spectral_image = axes[0].pcolormesh(
        time_edges, frequency_edges, display(spectrogram_normalised), **common
    )
    axes[0].set_title("Spectrogram")
    axes[0].set_ylabel("Frequency (Hz)")
    style = LANDSCAPE_FIGURE_STYLE
    for axis, surface, (_, time_reverse, frequency_reverse, start, _) in zip(
        axes[1:], surfaces, DIRECTIONS, strict=True
    ):
        axis.pcolormesh(time_edges, frequency_edges, display(surface), **common)
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
        axis.set_ylim(START_HZ, args.max_frequency)
        axis.set_xticks([0.5, 1, 1.5], ["0.5", "1", "1.5"])
        axis.set_xlim((0, DURATION_SECONDS) if args.staircase
                      else (time_edges[0], time_edges[-1]))
        ticks = [100, 1000]
        labels = [r"$10^2$", r"$10^3$"]
        axis.set_yticks(ticks, labels)
        axis.minorticks_off()
        axis.tick_params(length=2, width=0.5, pad=1.5)
    figure.canvas.draw()
    bounds = axes[-1].get_position()
    figure.supxlabel("Time (s)", x=(axes[0].get_position().x0 + bounds.x1) / 2,
                    y=0.06, fontsize=7)
    bar_axis = figure.add_axes([bounds.x1 + 0.014, bounds.y0, 0.012, bounds.height])
    colorbar = figure.colorbar(
        spectral_image, cax=bar_axis, orientation="vertical",
        ticks=np.linspace(args.db_floor, 0, 5) if args.decibels else [0, 0.25, 0.5, 0.75, 1],
    )
    colorbar.ax.tick_params(length=1.5, width=0.5, pad=1, labelsize=6)
    colorbar.set_label("Relative power (dB)" if args.decibels else "Normalised power",
                       fontsize=7, labelpad=9, rotation=270)
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
        "schema": "cumulative-surfaces-figure-v2",
        "target": target_description,
        "device": args.device,
        "torch_version": torch.__version__,
        "stft": {"n_fft": CEL_N_FFT, "hop": CEL_HOP, "center": False,
                 "window": "periodic Hann", "shape": list(power.shape)},
        "display": {"spectrogram": "power / peak STFT power",
                    "surfaces": "S_q/m; accumulation uses linear power",
                    "colour_scale": f"10 log10, [{args.db_floor:g},0] dB" if args.decibels else "linear [0,1]",
                    "frequency_axis": f"logarithmic, 20-{args.max_frequency:g} Hz; sums use all bins",
                    "colorbar": "one vertical shared scale; distinct reference powers",
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
