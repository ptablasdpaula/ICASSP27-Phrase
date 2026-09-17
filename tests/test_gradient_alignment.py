"""Whole-phrase alignment invariants independent of synthesis."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from report_gradient_alignment import alignment  # noqa: E402


def data(descent, positions=None, assignments=None, ties=None):
    descent = np.asarray(descent, dtype=float)
    n, _, events, _ = descent.shape
    return dict(
        gradients=-descent,
        candidates=np.zeros((n, events, 2)) if positions is None else positions,
        target=np.ones((events, 2)),
        assignments=np.tile(np.arange(events), (n, 1)) if assignments is None else assignments,
        ties=np.zeros(n, dtype=bool) if ties is None else ties,
    )


def test_cancellation_zero_and_exact_exclusions():
    source = data([[[[1, 1], [-2, -2]]], [[[0, 0], [0, 0]]]])
    result = alignment(source)
    np.testing.assert_allclose(result["event-cosine"], 0, atol=1e-15)
    np.testing.assert_array_equal(result["phrase-descent"], 0)
    assert result["phrase-cosine"][0, 0] < 0
    assert result["phrase-cosine"][1, 0] == 0
    source["candidates"][:] = 1
    assert all(np.isnan(v).all() for v in alignment(source).values())


def test_single_event_and_already_correct_coordinate():
    source = data([[[[1, 10]]]], positions=np.array([[[0, 1]]]))
    result = alignment(source)
    np.testing.assert_allclose(result["event-cosine"], result["phrase-cosine"])
    np.testing.assert_allclose(result["phrase-cosine"], 1 / np.sqrt(101))
    assert result["phrase-descent"][0, 0] == 1
    source["ties"][:] = True
    assert all(np.isnan(v).all() for v in alignment(source).values())


def test_permutation_and_nonfinite():
    source = data([[[[1, 2], [3, 4]]]])
    source["target"] = np.array([[2.0, 3.0], [4.0, 5.0]])
    before = alignment(source)
    source["target"] = source["target"][::-1]
    source["assignments"] = 1 - source["assignments"]
    for metric, values in alignment(source).items():
        np.testing.assert_allclose(values, before[metric])
    source["gradients"][0, 0, 0, 0] = np.nan
    with pytest.raises(FloatingPointError):
        alignment(source)


def test_extended_registry_and_single_event_projections():
    from extend_gradient_cardinality import EXTENDED_COLUMNS, extended_candidates, extra_targets

    registry = extra_targets()
    assert len(registry) == 65
    assert EXTENDED_COLUMNS == (
        (1, "joint"),
        (2, "joint"),
        (4, "joint"),
        (6, "joint"),
        (8, "joint"),
        (1, "pitch"),
        (2, "pitch"),
        (4, "pitch"),
        (1, "time"),
        (2, "time"),
        (4, "time"),
    )
    for (name, target), (other_name, other_target) in zip(registry, extra_targets(), strict=True):
        assert name == other_name
        np.testing.assert_array_equal(target, other_target)
        if len(target) > 1:
            assert np.diff(target[:, 1]).min() >= 0.05
        assert np.all((target[:, 0] >= 0) & (target[:, 0] <= 2))
        assert np.all((target[:, 1] >= 0.2) & (target[:, 1] <= 1.8))
        projected = extended_candidates(name, target)
        np.testing.assert_array_equal(projected["pitch"][..., 0], projected["joint"][..., 0])
        np.testing.assert_array_equal(projected["time"][..., 1], projected["joint"][..., 1])
        np.testing.assert_array_equal(
            projected["pitch"][..., 1], np.broadcast_to(target[:, 1], (256, len(target)))
        )
        np.testing.assert_array_equal(
            projected["time"][..., 0], np.broadcast_to(target[:, 0], (256, len(target)))
        )


def test_extended_render(tmp_path):
    from extend_gradient_cardinality import EXTENDED_COLUMNS
    from report_gradient_alignment import render

    render(tmp_path, "phrase-cosine", np.zeros((11, 11)), np.ones((11, 11)), EXTENDED_COLUMNS)
    assert (tmp_path / "phrase-cosine.pdf").stat().st_size > 1000
