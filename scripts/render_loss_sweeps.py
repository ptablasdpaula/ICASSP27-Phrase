"""Review-only 1D loss sweeps, without changing paper figures or shared losses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from icassp27_phrase.losses import (
    SOT_MSS_HOPS,
    SOT_MSS_WINDOWS,
    _stft,
    _wasserstein_frequency_rows,
    build_loss,
)
from icassp27_phrase.runtime import configure_reproducibility, require_df2_backend
from icassp27_phrase.synth import PhraseSynth

NAMES = (
    "waveform_l1",
    "linear_mss",
    "sot_published_composite",
    "waveform_mse",
    "smooth_mss",
    "linear_jtfot",
    "bidirectional_cumulative_energy",
)
LABELS = (r"$L_1$", "Linear MSS", "SOT", r"$L_2$", "Smooth MSS", "TFW", r"Ce$\mathcal{L}$")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#0072B2", "#D55E00", "#009E73", "#6A3D9A")
STYLES = ("-", "--", "-.", "-", "--", "-.", "-")


class LinearMSS:
    """Standalone unscaled linear-magnitude term of the registered SOT composite."""

    def __init__(self, target):
        self.windows = [
            torch.hann_window(n, periodic=True, dtype=target.dtype) for n in SOT_MSS_WINDOWS
        ]
        self.references = [
            _stft(target[None], n_fft=n, hop=h, window=w, center=True, pad_mode="reflect").abs()
            for n, h, w in zip(SOT_MSS_WINDOWS, SOT_MSS_HOPS, self.windows, strict=True)
        ]

    def __call__(self, audio):
        return sum(
            (_stft(audio, n_fft=n, hop=h, window=w, center=True, pad_mode="reflect").abs() - ref)
            .abs()
            .mean(dim=(-2, -1))
            for n, h, w, ref in zip(
                SOT_MSS_WINDOWS, SOT_MSS_HOPS, self.windows, self.references, strict=True
            )
        )


def qualify(synth, target, objectives):
    """Check standalone MSS against the actual composite minus its transport term."""
    probes = synth(
        torch.tensor([[100.0], [160.0], [250.0]], dtype=torch.float64),
        torch.tensor([[0.4], [1.0], [1.7]], dtype=torch.float64),
    )
    sot = objectives[2]
    x = (
        _stft(probes, n_fft=512, hop=64, window=sot.sot_window, center=True, pad_mode="reflect")
        .abs()
        .square()
        .transpose(-2, -1)
    )
    y = sot.target_power.expand(len(probes), -1, -1).transpose(-2, -1)
    mass = x.sum(-1, keepdim=True) + 1e-8
    transport = (
        _wasserstein_frequency_rows(
            (x / mass).reshape(-1, x.shape[-1]), (y / mass).reshape(-1, y.shape[-1]), sot.positions
        )
        .reshape(x.shape[:2])
        .mean(-1)
    )
    expected = (sot(probes) - transport) / 0.05
    actual = objectives[1](probes)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    self_losses = [float(obj(target[None])[0]) for obj in objectives]
    assert all(abs(v) < 1e-10 for v in self_losses), self_losses
    return {
        "linear_mss_component_agreement": True,
        "self_losses": dict(zip(NAMES, self_losses, strict=True)),
        "linear_mss_max_absolute_difference": float((actual - expected).abs().max()),
    }


def render(output, axes, values):
    spans = np.ptp(values, axis=-1)
    assert np.all(spans > 0)
    normalised = (values - values.min(axis=-1, keepdims=True)) / spans[..., None]
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

    def panel(ax, row, indices):
        ax.axvline(0, color="0.45", linestyle=":", linewidth=1, zorder=0)
        for index in indices:
            ax.plot(
                axes[row],
                normalised[row, index],
                color=COLORS[index],
                linestyle=STYLES[index],
                linewidth=1.25,
                label=LABELS[index],
            )
        ax.set(
            xlim=(axes[row, 0], axes[row, -1]),
            ylim=(-0.035, 1.07),
            xlabel="Onset displacement (s)" if row == 0 else r"Pitch displacement (octaves)",
            ylabel="Normalised loss",
            xticks=np.linspace(axes[row, 0], axes[row, -1], 5),
            yticks=[0, 0.5, 1],
        )
        ax.grid(axis="y", alpha=0.18)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.23),
            ncol=len(indices),
            frameon=False,
            fontsize=9,
            handlelength=2.2,
            columnspacing=1.2,
        )

    for stem, groups, width in [("comparison", [(0, 1, 2), (3, 4, 5)], 10), ("cel", [(6,)], 5)]:
        fig, axs = plt.subplots(2, len(groups), figsize=(width, 5.9), squeeze=False)
        for row in range(2):
            for column, group in enumerate(groups):
                panel(axs[row, column], row, group)
        fig.subplots_adjust(
            left=0.085 if len(groups) == 2 else 0.16,
            right=0.95,
            bottom=0.1,
            top=0.91,
            hspace=0.67,
            wspace=0.25,
        )
        for ext in ("pdf", "png"):
            fig.savefig(output / f"{stem}.{ext}", dpi=180)
        plt.close(fig)
    return normalised


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/loss-sweeps"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--row", type=int, choices=(0, 1))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    configure_reproducibility()
    torch.use_deterministic_algorithms(True)
    require_df2_backend("cpu")
    synth = PhraseSynth()
    axes = np.stack([np.linspace(-0.8, 0.8, 3201), np.linspace(-1, 1, 3201)])
    assert np.all(axes[:, 1600] == 0)
    cache = Path("results/loss-sweeps")
    cache.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    for path in [Path(__file__), *sorted(Path("src").glob("*.py"))]:
        digest.update(path.read_bytes())
    signature = digest.hexdigest()
    with torch.no_grad():
        target = synth(
            torch.tensor([[160.0]], dtype=torch.float64), torch.tensor([[1.0]], dtype=torch.float64)
        )[0]
        objectives = [
            LinearMSS(target) if name == "linear_mss" else build_loss(name, target)
            for name in NAMES
        ]
        checks = qualify(synth, target, objectives)
        print("Qualification passed", flush=True)
        values = np.empty((2, len(NAMES), 3201))
        for row, axis in enumerate(axes):
            if args.row is not None and row != args.row:
                continue
            shard = cache / f"row-{row}.npz"
            if shard.exists():
                with np.load(shard) as saved:
                    if str(saved["signature"]) == signature:
                        values[row] = saved["losses"]
                        print(f"Reused qualified source-matched row {row}", flush=True)
                        continue
            for start in range(0, len(axis), args.batch_size):
                displacement = torch.from_numpy(axis[start : start + args.batch_size]).reshape(
                    -1, 1
                )
                pitch = 160 * 2**displacement if row else torch.full_like(displacement, 160)
                onset = torch.ones_like(displacement) if row else 1 + displacement
                audio = synth(pitch.clamp(80.0, 320.0), onset.clamp(0.2, 1.8))
                for index, obj in enumerate(objectives):
                    values[row, index, start : start + len(displacement)] = obj(audio).numpy()
                if start % (args.batch_size * 20) == 0:
                    print(f"{'pitch' if row else 'onset'}: {start}/{len(axis)}", flush=True)
            np.savez_compressed(shard, losses=values[row], signature=signature)
    if args.row is not None:
        return
    assert np.isfinite(values).all() and values.min() >= -1e-12
    target_relative = np.abs(values[:, :, 1600]) / np.ptp(values, axis=-1)
    # Tiny batch-render roundoff can yield nonzero quantile-transport costs in Hz^2.
    assert target_relative.max() < 1e-8, target_relative
    checks["sweep_target_relative_to_range"] = target_relative.tolist()
    checks["sweep_target_raw_losses"] = values[:, :, 1600].tolist()
    normalised = render(args.output, axes, values)
    np.savez_compressed(
        args.output / "raw-sweeps.npz",
        displacement=axes,
        losses=values,
        normalised=normalised,
        names=np.array(NAMES),
    )
    with (args.output / "sweeps.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["sweep", "displacement", *NAMES])
        for row, axis in enumerate(axes):
            for index, displacement in enumerate(axis):
                writer.writerow(
                    [
                        "pitch_octaves" if row else "onset_seconds",
                        displacement,
                        *values[row, :, index],
                    ]
                )
    files = [Path(__file__), *Path("src").glob("*.py")]
    provenance = {
        "target": {"pitch_hz": 160, "onset_seconds": 1},
        "points_per_sweep": 3201,
        "losses": NAMES,
        "renderer": synth.provenance(),
        "torch_version": torch.__version__,
        "device": "cpu",
        "dtype": "float64",
        "qualification": checks,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "normalisation": "Independent min-max per loss per sweep; no smoothing.",
        "linear_mss": {
            "windows": SOT_MSS_WINDOWS,
            "hops": SOT_MSS_HOPS,
            "window": "periodic Hann",
            "comparison": "sum of mean absolute magnitude differences",
        },
    }
    provenance["artifact_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.output.iterdir()
        if p.suffix in (".pdf", ".png", ".npz", ".csv")
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("Complete", flush=True)


if __name__ == "__main__":
    main()
