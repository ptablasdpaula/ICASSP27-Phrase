"""Scientific invariants for the full-domain gradient assessment."""

import numpy as np
import pytest
from icassp27_phrase.gradient_assessment import candidates, matching, score, targets


def test_target_design_and_projected_lhs():
    registry = targets()
    assert len(registry) == 65
    np.testing.assert_array_equal(registry[0][1], [[1.0, 1.0]])
    for name, target in registry:
        if len(target) > 1:
            assert np.diff(target[:, 1]).min() >= 0.05
        design = candidates(name, target)
        joint = design["joint"]
        unit = (joint - [0, 0.2]) / [2, 1.6]
        assert np.all((unit >= 0) & (unit < 1))
        strata = np.floor(unit.reshape(256, -1) * 256).astype(int)
        np.testing.assert_array_equal(
            np.sort(strata, axis=0), np.broadcast_to(np.arange(256)[:, None], strata.shape)
        )
        np.testing.assert_array_equal(joint, candidates(name, target)["joint"])
        assert not np.array_equal(joint, candidates(name, target, repeat=1)["joint"])
        if len(target) > 1:
            np.testing.assert_array_equal(design["pitch"][..., 0], joint[..., 0])
            np.testing.assert_array_equal(design["time"][..., 1], joint[..., 1])
            np.testing.assert_array_equal(
                design["pitch"][..., 1], np.broadcast_to(target[:, 1], (256, len(target)))
            )
            np.testing.assert_array_equal(
                design["time"][..., 0], np.broadcast_to(target[:, 0], (256, len(target)))
            )


def test_equal_octave_second_cost_and_unrestricted_conditional_matching():
    # Symmetry under exchanging octave and second coordinates, including ties.
    target = np.array([[0.0, 0.2], [1.0, 1.2]])
    candidate = np.array([[1.0, 0.2], [0.0, 1.2]])
    _, tied = matching(candidate, target)
    assert tied  # retaining pitch or retaining onset has identical cost
    target = np.array([[0.0, 0.4], [2.0, 1.6]])
    candidate = np.array([[2.0, 0.4], [0.0, 1.6]])  # onsets were initially exact
    assignment, tied = matching(candidate, target)
    assert not tied
    np.testing.assert_array_equal(assignment, [1, 0])
    exchanged, exchanged_tie = matching(candidate[:, ::-1], target[:, ::-1])
    np.testing.assert_array_equal(exchanged, assignment)
    assert exchanged_tie == tied


def test_componentwise_score_does_not_hide_wrong_pitch():
    target = np.array([[1.0, 1.0]])
    positions = np.array([[[0.0, 0.0]], [[0.0, 0.0]], [[1.0, 1.0]]])
    gradients = np.array([[[[1.0, -2.0]]], [[[0.0, 0.0]]], [[[0.0, 0.0]]]])
    result = score(gradients, positions, target, np.zeros((3, 1), dtype=int), np.zeros(3, bool))
    assert result["dot_directed"][0, 0] == 1
    assert result["both_directed"][0, 0] == 0
    assert result["both_directed"][1, 0] == 0
    assert np.isnan(result["both_directed"][2, 0])


def test_matching_scoring_permutation_and_ties():
    target = np.array([[0.2, 0.3], [1.7, 1.6]])
    positions = target[::-1][None] + 0.01
    assignment, tied = matching(positions[0], target)
    gradient = (positions - target[assignment])[..., None, :, :]
    result = score(gradient, positions, target, assignment[None], np.array([tied]))
    assert result["both_directed"].item() == 1
    permuted = score(
        gradient[:, :, ::-1], positions[:, ::-1], target, assignment[None, ::-1], np.array([tied])
    )
    assert permuted["both_directed"].item() == 1
    excluded = score(gradient, positions, target, assignment[None], np.array([True]))
    assert np.isnan(excluded["both_directed"]).all()


def test_nonfinite_gradients_fail_instead_of_changing_denominators():
    with pytest.raises(FloatingPointError):
        score(
            np.full((1, 1, 1, 2), np.nan),
            np.zeros((1, 1, 2)),
            np.ones((1, 2)),
            np.zeros((1, 1), dtype=int),
            np.zeros(1, bool),
        )


def test_target_averaging_and_checkpoint_validation(tmp_path):
    import importlib.util
    from pathlib import Path

    from icassp27_phrase.gradient_assessment import NAMES, SCHEMA, signature

    script = Path(__file__).resolve().parents[1] / "scripts/assess_gradients.py"
    spec = importlib.util.spec_from_file_location("assessment_runner", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Different eligible counts must not give one target more population weight.
    target_means = [module.finite_mean([[1.0], [np.nan]]), module.finite_mean([[0.0], [0.0]])]
    assert np.mean(target_means) == 0.5
    name, target = targets()[0]
    positions = candidates(name, target, count=4)["joint"]
    path = tmp_path / "checkpoint.npz"
    data = dict(
        schema=SCHEMA,
        signature=signature()[0],
        names=np.array(NAMES),
        candidates=positions,
        target=target,
        losses=np.zeros((4, len(NAMES))),
        gradients=np.zeros((4, len(NAMES), 1, 2)),
        assignments=np.zeros((4, 1), int),
        ties=np.zeros(4, bool),
    )
    np.savez(path, **data)
    loaded = module.checked_data(path)
    np.testing.assert_array_equal(loaded["candidates"], positions)
    np.savez(path, **{**data, "signature": "stale"})
    with pytest.raises(ValueError, match="stale"):
        module.checked_data(path)
    np.savez(path, **{**data, "gradients": np.zeros((3, len(NAMES), 1, 2))})
    with pytest.raises(ValueError, match="incomplete/invalid"):
        module.checked_data(path)
