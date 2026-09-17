"""Add pointwise magnitude sweeps using the four-direction CeL STFT settings."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from icassp27_phrase.losses import BidirectionalCumulativeEnergyDistance, _stft
from icassp27_phrase.runtime import configure_reproducibility, require_df2_backend
from icassp27_phrase.synth import PhraseSynth


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = Path("docs/loss-sweeps")
    source = output / "raw-sweeps.npz"
    previous = json.loads((output / "provenance.json").read_text())
    assert sha(source) == previous["artifact_sha256"][source.name]
    axes = np.load(source)["displacement"]
    torch.set_num_threads(1)
    configure_reproducibility()
    torch.use_deterministic_algorithms(True)
    require_df2_backend("cpu")
    synth = PhraseSynth()
    assert synth.provenance() == previous["renderer"]
    values = np.empty_like(axes)
    with torch.no_grad():
        target = synth(
            torch.tensor([[160.0]], dtype=torch.float64), torch.tensor([[1.0]], dtype=torch.float64)
        )[0]
        cel = BidirectionalCumulativeEnergyDistance(target)

        def magnitude(audio):
            return _stft(
                audio,
                n_fft=cel.n_fft,
                hop=cel.hop,
                window=cel.window,
                center=False,
                pad_mode="constant",
            ).abs()

        reference = magnitude(target[None])
        torch.testing.assert_close(reference.square(), cel._power(target[None]), rtol=0, atol=0)
        assert float((reference - magnitude(target[None])).abs().mean()) == 0
        for row, axis in enumerate(axes):
            for start in range(0, len(axis), 16):
                d = torch.from_numpy(axis[start : start + 16]).reshape(-1, 1)
                pitch = 160 * 2**d if row else torch.full_like(d, 160)
                onset = torch.ones_like(d) if row else 1 + d
                audio = synth(pitch.clamp(80, 320), onset.clamp(0.2, 1.8))
                values[row, start : start + len(d)] = (
                    (magnitude(audio) - reference).abs().mean((-2, -1)).numpy()
                )
                if start % 640 == 0:
                    print(f"row {row}: {start}/{len(axis)}", flush=True)
    assert np.isfinite(values).all() and values.min() >= 0
    residual = values[:, 1600] / np.ptp(values, axis=-1)
    assert residual.max() < 1e-12
    normalised = (values - values.min(axis=-1, keepdims=True)) / np.ptp(values, axis=-1)[:, None]
    artifact = output / "single-stft-sweeps.npz"
    np.savez_compressed(artifact, displacement=axes, losses=values, normalised=normalised)
    provenance = {
        "objective": "mean absolute pointwise STFT magnitude difference",
        "n_fft": cel.n_fft,
        "hop": cel.hop,
        "window": "periodic Hann",
        "center": False,
        "sample_rate": synth.sample_rate,
        "renderer": synth.provenance(),
        "source_sweeps_sha256": sha(source),
        "artifact_sha256": sha(artifact),
        "source_sha256": {
            str(p): sha(p) for p in [Path(__file__), *sorted(Path("src").glob("*.py"))]
        },
        "qualification": {
            "stft_power_matches_cel_exactly": True,
            "self_loss": 0,
            "sweep_target_relative_to_range": residual.tolist(),
        },
        "normalisation": "Independent min-max per sweep; no smoothing or downsampling",
    }
    (output / "single-stft.provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("Complete", flush=True)


if __name__ == "__main__":
    main()
