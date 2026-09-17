"""Standalone CeL panel from the current 2.048-padding landscape data."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from scipy.ndimage import zoom

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper/figures/loss_landscapes.padding2048.npz"
OUTPUT = ROOT / "docs/cel-landscape"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with np.load(SOURCE) as saved:
        surface = saved["surfaces"][5]
        directions = saved["directions"][5]
        count = int(saved["counts"][5])
        grid = saved["grid"].reshape(25, 25, 2)
        renderer = json.loads(str(saved["renderer_json"]))
    assert surface.shape == (25, 25) and directions.shape == (120, 2)
    assert renderer["exciter"]["fourier_fft_length"] == 16384
    positions = np.array(
        [grid[i, j] for i in range(2, 23, 2) for j in range(2, 23, 2) if (i, j) != (12, 12)]
    )
    assert np.allclose(np.linalg.norm(directions, axis=1), 1)
    assert np.isfinite(surface).all() and surface.argmin() == 12 * 25 + 12
    measured = int(np.count_nonzero(np.sum(directions * (0.5 - positions), axis=1) > 0))
    assert measured == count == 120
    normalised = (surface - surface.min()) / np.ptp(surface)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(4.5, 4.15))
    ax = fig.add_axes((0.15, 0.15, 0.69, 0.75))
    # Same 25-to-180 bilinear display interpolation as the original figure.
    im = ax.imshow(
        zoom(normalised, 180 / 25, order=1),
        origin="lower",
        extent=(-0.8, 0.8, -1, 1),
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    for point, direction in zip(positions, directions, strict=True):
        centre, vector = point[::-1], direction[::-1]
        ax.add_patch(
            FancyArrowPatch(
                centre - 0.026 * vector,
                centre + 0.026 * vector,
                transform=ax.transAxes,
                arrowstyle="-|>",
                mutation_scale=5.5,
                linewidth=0.55,
                color="white",
                shrinkA=0,
                shrinkB=0,
                clip_on=True,
                zorder=4,
            )
        )
    ax.plot(0, 0, marker="+", color="#fff59d", markersize=11, markeredgewidth=1.8, zorder=5)
    ax.text(
        0.97,
        0.97,
        f"{100 * count / 120:.1f}%",
        transform=ax.transAxes,
        color="white",
        ha="right",
        va="top",
        fontsize=10,
        zorder=6,
    )
    ax.set(
        title=r"Ce$\mathcal{L}$",
        xlabel="Time shift (s)",
        ylabel="Frequency shift (octaves)",
        xticks=[-0.8, 0, 0.8],
        yticks=[-1, 0, 1],
    )
    ax.set_xticklabels(["−0.8", "0", "+0.8"])
    ax.set_yticklabels(["−1", "0", "+1"])
    ax.get_xticklabels()[0].set_horizontalalignment("left")
    ax.get_xticklabels()[-1].set_horizontalalignment("right")
    bar = fig.colorbar(im, cax=fig.add_axes((0.875, 0.15, 0.025, 0.75)), ticks=[0, 1])
    bar.set_label("Normalised loss", labelpad=1)
    for extension in ("png", "pdf"):
        path = OUTPUT / f"cel-landscape.{extension}"
        fig.savefig(path, dpi=200)
        print(path, flush=True)
    plt.close(fig)
    report = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "loss": "bidirectional_cumulative_energy",
        "directions": "all four diagonals",
        "target": {"pitch_hz": 160, "onset_seconds": 1},
        "renderer": renderer,
        "surface_grid": [25, 25],
        "displayed_gradients": 120,
        "target_directed_gradients": measured,
        "normalisation": "surface min-max",
        "display_interpolation": "bilinear 25 to 180",
        "gradient_coordinates": "unit square in pitch and onset; arrows are unit descent",
        "artifacts": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in OUTPUT.iterdir()
            if p.suffix in (".png", ".pdf")
        },
    }
    (OUTPUT / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
