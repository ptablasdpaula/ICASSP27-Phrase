"""Independent checks for the experimental orthogonal extension."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
from icassp27_phrase.losses import CumulativeEnergyDistance

spec = importlib.util.spec_from_file_location(
    'eight', Path(__file__).resolve().parents[1] / 'scripts/screen_eight_directions.py')
eight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eight)


class Grid(CumulativeEnergyDistance):
    def _power(self, rows):
        return rows.reshape(-1, 3, 3)


def test_axis_sums_and_quadrature_against_explicit_slices():
    target = torch.arange(1, 10, dtype=torch.float64)
    candidate = target.flip(0)
    loss = eight.AxisLoss(Grid(target))
    actual = loss.terms(candidate[None])[0].numpy()
    p, c = target.numpy().reshape(3, 3), candidate.numpy().reshape(3, 3)
    f = np.log2(np.maximum(np.arange(3) * 4000 / 256, 20) / 20)
    t = np.arange(3) * 64 / 4000
    fw = (np.r_[np.diff(f), 0], np.r_[0, np.diff(f)])
    tw = (np.r_[np.diff(t), 0], np.r_[0, np.diff(t)])
    ordinary, weighted = [], []
    for direction in range(4):
        error = np.zeros((3, 3))
        for k in range(3):
            for n in range(3):
                fs = (slice(None, k + 1) if direction == 0 else slice(k, None)
                      if direction == 1 else k)
                ts = (slice(None, n + 1) if direction == 2 else slice(n, None)
                      if direction == 3 else n)
                error[k, n] = (np.sqrt(c[fs, ts].sum() / p.sum())
                               - np.sqrt(p[fs, ts].sum() / p.sum()))
        wf = fw[direction] if direction < 2 else (fw[0] + fw[1]) / 2
        wt = tw[direction - 2] if direction >= 2 else (tw[0] + tw[1]) / 2
        weight = wf[:, None] * wt[None, :]
        ordinary.append(np.sqrt(np.mean(error ** 2)))
        weighted.append(np.sqrt((error ** 2 * weight).sum() / weight.sum()))
    np.testing.assert_allclose(actual, ordinary + weighted, atol=1e-15)
    np.testing.assert_allclose(loss.weights.square().sum((-1, -2)), np.ones(4))


def test_mixed_subset_gradients_and_finite_differences():
    target = torch.arange(1, 10, dtype=torch.float64)
    candidate = torch.tensor([9., 7., 8., 4., 2., 1., 3., 5., 6.],
                             dtype=torch.float64, requires_grad=True)
    canonical = Grid(target)
    axis = eight.AxisLoss(canonical)

    def terms(x):
        d = canonical.directional_distances(x).flatten()
        a = axis.terms(x[None])[0]
        return torch.cat((d[:4], a[:4], d[4:], a[4:]))

    values = terms(candidate)
    gradients = torch.stack([torch.autograd.grad(v, candidate, retain_graph=True)[0]
                             for v in values])
    assert len(eight.NAMES) == len(set(eight.NAMES)) == 510
    for weighted in (False, True):
        for mask in (17, 48, 192, 255):
            indices = [i + (8 if weighted else 0) for i in range(8) if mask & (1 << i)]
            direct = torch.autograd.grad(values[indices].mean(), candidate, retain_graph=True)[0]
            torch.testing.assert_close(direct, gradients[indices].mean(0), atol=1e-14, rtol=1e-12)
            for coordinate in (0, 4, 8):
                step = torch.zeros_like(candidate)
                step[coordinate] = 1e-5
                numerical = (terms(candidate + step)[indices].mean()
                             - terms(candidate - step)[indices].mean()) / 2e-5
                torch.testing.assert_close(direct[coordinate], numerical, atol=1e-10, rtol=1e-7)
    same = target.clone().requires_grad_()
    value = axis.terms(same[None]).sum()
    assert value == 0
    assert torch.equal(torch.autograd.grad(value, same)[0], torch.zeros_like(same))
