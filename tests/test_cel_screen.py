from __future__ import annotations

import numpy as np
import pytest
import torch
from icassp27_phrase.cel_screen import candidates, design, matching, score, subset_matrix
from icassp27_phrase.config import ExciterConfig
from icassp27_phrase.losses import CEL_NAMES, CumulativeEnergyDistance, build_loss


def test_new_padding_and_independent_design():
    assert ExciterConfig().fourier_fft_length == 65536
    targets = design()
    assert len(targets) == 173
    for target in targets:
        positions, kinds = candidates(target)
        assert positions.shape[1:] == target.coordinates.shape
        assert np.all((positions >= 0) & (positions <= 1))
        assert (kinds == "simultaneous").sum() == 32
        if target.profile == "independent":
            assert np.all(np.diff(target.coordinates[:, 1]) * 1.6 >= 0.05)
    np.testing.assert_array_equal(targets[-1].coordinates, design()[-1].coordinates)


@pytest.mark.parametrize(
    "weighted,legacy", [(False, "bidirectional_cumulative_energy"), (True, "log_quadrature_bicul")]
)
def test_legacy_value_and_gradient_equivalence(weighted, legacy):
    generator = torch.Generator().manual_seed(17)
    target = torch.randn(2048, generator=generator, dtype=torch.float64)
    candidate = torch.randn(2, 2048, generator=generator, dtype=torch.float64, requires_grad=True)
    old = build_loss(legacy, target)(candidate)
    new = CumulativeEnergyDistance(target, log_weighing=weighted)(candidate)
    torch.testing.assert_close(new, old, rtol=1e-12, atol=1e-12)
    (a,) = torch.autograd.grad(old.sum(), candidate)
    (b,) = torch.autograd.grad(new.sum(), candidate)
    torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)


def test_all_subsets_and_exact_match_gradients():
    generator = torch.Generator().manual_seed(18)
    target = torch.randn(2048, generator=generator, dtype=torch.float64)
    candidate = torch.randn(2048, generator=generator, dtype=torch.float64, requires_grad=True)
    elementary = CumulativeEnergyDistance(target).directional_distances(candidate).flatten()
    matrix = torch.tensor(subset_matrix(), dtype=torch.float64)
    expected = matrix @ elementary
    for i, name in enumerate(CEL_NAMES):
        actual = build_loss(name, target)(candidate)
        torch.testing.assert_close(actual, expected[i])
        (a,) = torch.autograd.grad(actual, candidate)
        (b,) = torch.autograd.grad(expected[i], candidate, retain_graph=True)
        torch.testing.assert_close(a, b)
    same = target.clone().requires_grad_()
    value = CumulativeEnergyDistance(target, log_weighing=True)(same)
    (grad,) = torch.autograd.grad(value, same)
    assert value == 0
    assert torch.equal(grad, torch.zeros_like(grad))
    with pytest.raises(ValueError):
        CumulativeEnergyDistance(target, directions=())


def test_matching_ties_and_permutation():
    target = np.array([[0.2, 0.3], [0.7, 0.8]])
    candidate = target[::-1] + 0.01
    assignment, tied = matching(candidate, target)
    assert not tied
    np.testing.assert_array_equal(assignment, [1, 0])
    gradient = (candidate - target[assignment])[None, None]
    result = score(gradient, (target[assignment] - candidate)[None])
    assert result["joint_directed"].item() == 1
    permuted = score(gradient[:, :, ::-1], (target[assignment] - candidate)[None, ::-1])
    np.testing.assert_allclose(permuted["joint_cosine"], result["joint_cosine"])
    _, tied = matching(np.full((2, 2), 0.5), target)
    assert tied


def test_joint_alignment_does_not_hide_wrong_axis_and_zero():
    delta = np.array([[[1.0, 1.0], [0.0, 1.0]]])
    gradient = np.array([[[[1.0, -2.0], [0.0, 0.0]]]])
    result = score(gradient, delta)
    assert result["joint_directed"].item() == 0.5
    assert result["pitch_directed"].item() == 0
    assert result["onset_directed"].item() == 0.5
    assert result["zero_gradient"].item() == 0.5
    assert result["phrase_directed"].item() == 1


def test_named_directions_against_explicit_rectangles():
    class PowerGrid(CumulativeEnergyDistance):
        def _power(self, rows):
            return rows.reshape(-1, 3, 3)

    target = torch.arange(1, 10, dtype=torch.float64)
    candidate = torch.flip(target, (0,))
    actual = PowerGrid(target).directional_distances(candidate)[0].numpy()
    expected = []
    p, c = target.numpy().reshape(3, 3), candidate.numpy().reshape(3, 3)
    for reverse_t, reverse_f in ((False, False), (False, True), (True, False), (True, True)):
        errors = []
        for k in range(3):
            for n in range(3):
                fs = slice(k, None) if reverse_f else slice(None, k + 1)
                ts = slice(n, None) if reverse_t else slice(None, n + 1)
                errors.append(
                    np.sqrt(c[fs, ts].sum() / p.sum()) - np.sqrt(p[fs, ts].sum() / p.sum())
                )
        expected.append(np.sqrt(np.mean(np.square(errors))))
    np.testing.assert_allclose(actual, expected, atol=1e-15)


def test_report_weights_targets_equally(tmp_path, monkeypatch):
    import csv
    import importlib.util
    from pathlib import Path

    from icassp27_phrase.cel_screen import ScreenTarget, source_hash

    script = Path(__file__).resolve().parents[1] / "scripts/screen_cel_gradients.py"
    spec = importlib.util.spec_from_file_location("screen_report", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    targets = [ScreenTarget(name, "independent", np.array([[0.8, 0.8]])) for name in ("a", "b")]
    monkeypatch.setattr(module, "design", lambda: targets)
    monkeypatch.setattr(module, "heatmaps", lambda *args: None)
    monkeypatch.setattr(module, "write_report", lambda *args: None)
    raw = tmp_path / "raw"
    raw.mkdir()
    for target, signs in zip(targets, ([1, -1], [-1]), strict=True):
        gradients = np.array(signs)[:, None, None, None] * np.ones((len(signs), 8, 1, 2))
        gradients[:, 7] = np.nan  # Only subsets selecting this term should fail.
        np.savez(
            raw / f"{target.name}.npz",
            source_hash=source_hash(),
            device="cpu",
            target=target.coordinates,
            candidates=np.full((len(signs), 1, 2), 0.2),
            assignments=np.zeros((len(signs), 1), dtype=int),
            elementary_gradients=gradients,
            tied=np.zeros(len(signs), dtype=bool),
            kinds=np.array(["simultaneous"] * len(signs)),
        )
    module.report(raw, tmp_path / "report")
    with (tmp_path / "report/summary.csv").open() as f:
        row = next(
            r
            for r in csv.DictReader(f)
            if r["variant"] == "cel_15" and r["metric"] == "joint_directed"
        )
    assert int(row["phrases"]) == 2
    assert float(row["mean"]) == 0.75
    assert float(row["median"]) == 0.75
    import json

    metadata = json.loads((tmp_path / "report/provenance.json").read_text())
    assert metadata["quality"]["nonfinite_variant_candidates"] == 24
