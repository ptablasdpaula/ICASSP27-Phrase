"""Recompute Figure 3 BiCuL arrows for an alternative onset FFT length."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from icassp27_phrase.config import ExciterConfig
from icassp27_phrase.losses import build_loss
from icassp27_phrase.runtime import configure_reproducibility, require_df2_backend
from icassp27_phrase.synth import PhraseSynth
from render_loss_landscapes import controls, displayed_coordinates, target_direction_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fft-length", type=int, default=32768)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_reproducibility()
    require_df2_backend(args.device)
    synth = PhraseSynth(exciter_config=ExciterConfig(fourier_fft_length=args.fft_length)).to(
        args.device
    )
    with torch.no_grad():
        centre = torch.tensor([[0.5, 0.5]], dtype=torch.float64, device=args.device)
        target = synth(*controls(centre))[0]
    metric = build_loss("bidirectional_cumulative_energy", target)
    positions = displayed_coordinates()
    gradients, losses = [], []
    for begin in range(0, 120, 8):
        coordinates = torch.tensor(
            positions[begin : begin + 8],
            dtype=torch.float64,
            device=args.device,
            requires_grad=True,
        )
        values = metric.distances(synth(*controls(coordinates)))
        (grad,) = torch.autograd.grad(values.sum(), coordinates)
        gradients.extend(grad.detach().cpu().tolist())
        losses.extend(values.detach().cpu().tolist())
        print(f"{begin + len(values)}/120", flush=True)
    gradients = np.asarray(gradients)
    if not np.isfinite(gradients).all() or not np.isfinite(losses).all():
        raise FloatingPointError("Nonfinite diagnostic values")
    norms = np.linalg.norm(gradients, axis=1, keepdims=True)
    unit = np.divide(-gradients, norms, out=np.zeros_like(gradients), where=norms > 0)
    dots = np.sum(unit * (0.5 - positions), axis=1)
    result = {
        **target_direction_summary(positions, unit),
        "device": args.device,
        "torch_version": torch.__version__,
        "onset_fft_length": args.fft_length,
        "padding_ratio": args.fft_length / 8000,
        "target_f0_hz": 160.0,
        "target_onset_seconds": 1.0,
        "criterion": (
            "dot(unit negative gradient, target minus candidate) > 0 "
            "in normalized pitch/onset coordinates"
        ),
        "renderer": synth.provenance(),
        "minimum_target_dot": float(dots.min()),
        "zero_gradient_count": int(np.count_nonzero(norms == 0)),
        "coordinates_pitch_onset": positions.tolist(),
        "gradients_pitch_onset": gradients.tolist(),
        "losses": losses,
        "target_dots": dots.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in [
                    "target_directed_count",
                    "displayed_non_target_gradients",
                    "minimum_target_dot",
                    "zero_gradient_count",
                    "padding_ratio",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
