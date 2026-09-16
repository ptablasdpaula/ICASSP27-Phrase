"""Fresh six-objective Figure 3 at the current synthesis defaults."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from icassp27_phrase.losses import build_loss
from icassp27_phrase.runtime import configure_reproducibility, require_df2_backend
from icassp27_phrase.synth import PhraseSynth
from render_loss_landscapes import (
    BASE_LOSS_NAMES,
    PAPER_LOSS_LABELS,
    compute_directions,
    controls,
    coordinate_grids,
    render_figure,
)
from scipy.ndimage import zoom


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_reproducibility()
    require_df2_backend(args.device)
    synth = PhraseSynth().to(args.device)
    with torch.no_grad():
        x = torch.tensor([[0.5, 0.5]], dtype=torch.float64, device=args.device)
        target = synth(*controls(x))[0]
    directions, counts = compute_directions(
        synth, target, device=torch.device(args.device), chunk_size=8
    )
    print("counts", counts.tolist(), flush=True)
    grid = coordinate_grids()[2]
    metrics = [build_loss(name, target) for name in BASE_LOSS_NAMES]
    surfaces = []
    with torch.no_grad():
        for begin in range(0, len(grid), 32):
            x = torch.tensor(grid[begin : begin + 32], dtype=torch.float64, device=args.device)
            audio = synth(*controls(x))
            surfaces.append(np.stack([m(audio).cpu().numpy() for m in metrics]))
    surfaces = np.concatenate(surfaces, axis=1).reshape(6, 25, 25)
    if not np.isfinite(surfaces).all():
        raise FloatingPointError("nonfinite surface")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb = []
    for surface in surfaces:
        normalised = (surface - surface.min()) / (surface.max() - surface.min())
        expanded = zoom(normalised[::-1], 180 / 25, order=1)
        rgb.append((plt.get_cmap("magma")(expanded)[..., :3] * 255).astype(np.uint8))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render_figure(
        np.stack(rgb),
        directions,
        counts,
        args.output.with_suffix(".pdf"),
        labels=(*PAPER_LOSS_LABELS[:5], r"Ce$\mathcal{L}$ (four directions)"),
    )
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        surfaces=surfaces,
        directions=directions,
        counts=counts,
        grid=grid,
        renderer_json=json.dumps(synth.provenance()),
        device=args.device,
    )
    result = {
        "losses": BASE_LOSS_NAMES,
        "counts": counts.tolist(),
        "device": args.device,
        "renderer": synth.provenance(),
        "torch_version": torch.__version__,
        "surface_grid": [25, 25],
        "displayed_gradients": 120,
        "note": "Fresh computation; historical caches were not reused.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
