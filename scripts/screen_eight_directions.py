"""Extend the frozen CeL gradient screen to all 255 eight-direction subsets."""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from icassp27_phrase.cel_screen import candidates, design, source_hash
from icassp27_phrase.losses import CumulativeEnergyDistance, reverse_cumsum
from icassp27_phrase.runtime import configure_reproducibility, require_df2_backend
from icassp27_phrase.synth import PhraseSynth

DIRECTIONS = ("right_up", "right_down", "left_up", "left_down", "up", "down", "right", "left")
NAMES = tuple(f"cel8_{mask:03d}{'_lw' if lw else ''}"
              for lw in (False, True) for mask in range(1, 256))


def signature():
    return hashlib.sha256((source_hash() + Path(__file__).read_text()).encode()).hexdigest()


def axis_surfaces(power):
    return torch.stack((power.cumsum(-2), reverse_cumsum(power, -2),
                        power.cumsum(-1), reverse_cumsum(power, -1)), dim=1)


class AxisLoss:
    """Axis-only sums, global target mass; optional log-frequency quadrature.

    On the unaccumulated axis, use trapezoidal (mean forward/backward)
    quadrature widths, since there is no accumulation orientation there.
    """
    def __init__(self, canonical):
        self.canonical = canonical
        power = canonical._power(canonical.target[None])
        mass = power.sum().clamp_min(torch.finfo(power.dtype).tiny)
        self.mass = mass
        self.reference = (axis_surfaces(power) / mass).clamp_min(1e-12).sqrt().detach()
        w = canonical.sqrt_weights.square()
        self.weights = torch.stack(((w[0] + w[2]) / 2, (w[1] + w[3]) / 2,
                                    (w[0] + w[1]) / 2, (w[2] + w[3]) / 2)).sqrt()

    def terms(self, audio):
        power = self.canonical._power(audio)
        error = (axis_surfaces(power) / self.mass).clamp_min(1e-12).sqrt() - self.reference
        ordinary = torch.linalg.vector_norm(error.flatten(2), dim=-1) / np.sqrt(
            power.shape[-1] * power.shape[-2])
        weighted = torch.linalg.vector_norm((error * self.weights).flatten(2), dim=-1)
        return torch.cat((ordinary, weighted), dim=-1)


def compute(index, old_root, output, chunk_size):
    target = design()[index]
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{target.name}.npz"
    old_path = old_root / path.name
    old_sha = hashlib.sha256(old_path.read_bytes()).hexdigest()
    if path.exists():
        with np.load(path) as saved:
            assert str(saved['signature']) == signature()
            assert str(saved['old_sha256']) == old_sha
        return path
    with np.load(old_path) as saved:
        old = dict(saved)
    assert str(old['source_hash']) == source_hash(), 'Archived diagonal source has changed'
    assert str(old['device']) == 'cpu'
    positions, kinds = candidates(target)
    np.testing.assert_array_equal(positions, old['candidates'])
    np.testing.assert_array_equal(kinds, old['kinds'])
    np.testing.assert_array_equal(target.coordinates, old['target'])
    synth = PhraseSynth()
    coords = torch.tensor(target.coordinates, dtype=torch.float64)
    with torch.no_grad():
        audio = synth(80 * 4 ** coords[None, :, 0], .2 + 1.6 * coords[None, :, 1])[0]
    canonical = CumulativeEnergyDistance(audio)
    objective = AxisLoss(canonical)
    assert objective.terms(audio[None]).abs().max() == 0
    values, gradients = [], []
    for start in range(0, len(positions), chunk_size):
        x = torch.tensor(positions[start:start + chunk_size], dtype=torch.float64,
                         requires_grad=True)
        rendered = synth(80 * 4 ** x[..., 0], .2 + 1.6 * x[..., 1])
        terms = objective.terms(rendered)
        # Independently reproduce archived diagonal gradients on the first batch.
        if start == 0:
            diagonal = canonical.directional_distances(rendered).flatten(1)
            np.testing.assert_allclose(diagonal.detach(), old['elementary_losses'][:len(x)],
                                       rtol=1e-10, atol=1e-12)
            for i in range(8):
                g = torch.autograd.grad(diagonal[:, i].sum(), x, retain_graph=True)[0]
                np.testing.assert_allclose(g.detach(), old['elementary_gradients'][:len(x), i],
                                           rtol=1e-8, atol=1e-10)
        grads = [torch.autograd.grad(terms[:, i].sum(), x, retain_graph=i < 7)[0]
                 .detach().numpy() for i in range(8)]
        values.append(terms.detach().numpy())
        gradients.append(np.stack(grads, 1))
    axis_values, axis_gradients = np.concatenate(values), np.concatenate(gradients)
    # Order: eight uniform directions, then the same eight with Log-Weighing.
    losses = np.concatenate((old['elementary_losses'][:, :4], axis_values[:, :4],
                             old['elementary_losses'][:, 4:], axis_values[:, 4:]), axis=1)
    grads = np.concatenate((old['elementary_gradients'][:, :4], axis_gradients[:, :4],
                            old['elementary_gradients'][:, 4:], axis_gradients[:, 4:]), axis=1)
    assert np.isfinite(losses).all() and np.isfinite(grads).all()
    old.update(elementary_losses=losses, elementary_gradients=grads,
               signature=signature(), old_sha256=old_sha, directions=np.array(DIRECTIONS),
               schema='cel-eight-direction-screen-v1')
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **old)
    temporary.replace(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-root', type=Path, default=Path('results/cel/raw-cluster'))
    parser.add_argument('--output', type=Path, default=Path('results/eight-direction-screen/raw'))
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--index', type=int)
    parser.add_argument('--chunk-size', type=int, default=2)
    args = parser.parse_args()
    configure_reproducibility()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    require_df2_backend('cpu')
    indices = ([args.index] if args.index is not None
               else range(args.start, len(design()), args.stride))
    for index in indices:
        start = time.perf_counter()
        path = compute(index, args.old_root, args.output, args.chunk_size)
        print(index, path.name, round(time.perf_counter() - start, 2), flush=True)


if __name__ == '__main__':
    main()
