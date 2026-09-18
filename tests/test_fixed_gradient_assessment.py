"""Invariants for the paper's fixed LHS gradient design."""

import numpy as np
from icassp27_phrase.fixed_gradient_assessment import (
    CANDIDATES_PER_TARGET,
    CARDINALITIES,
    COLUMNS,
    TARGETS_PER_CARDINALITY,
    candidates,
    design_metadata,
    target_unit_design,
    targets,
)


def assert_latin(unit: np.ndarray) -> None:
    count = len(unit)
    strata = np.floor(unit.reshape(count, -1) * count).astype(int)
    np.testing.assert_array_equal(
        np.sort(strata, axis=0), np.broadcast_to(np.arange(count)[:, None], strata.shape)
    )


def test_fixed_target_design_is_reproducible_lhs_and_spaced():
    registry = targets()
    assert len(registry) == len(CARDINALITIES) * TARGETS_PER_CARDINALITY
    for events in CARDINALITIES:
        unit = target_unit_design(events)
        assert_latin(unit)
        np.testing.assert_array_equal(unit, target_unit_design(events))
        rows = [target for _, target in registry if len(target) == events]
        assert len(rows) == TARGETS_PER_CARDINALITY
        for target in rows:
            assert np.all((0 <= target[:, 0]) & (target[:, 0] < 2))
            assert np.all((0.2 <= target[:, 1]) & (target[:, 1] <= 1.8))
            if events > 1:
                assert np.diff(target[:, 1]).min() >= 0.05 - 1e-12


def test_every_target_has_one_paired_candidate_lhs_and_projections():
    for name, target in targets():
        design = candidates(name, target)
        joint = design["joint"]
        unit = (joint - [0, 0.2]) / [2, 1.6]
        assert_latin(unit)
        np.testing.assert_array_equal(joint, candidates(name, target)["joint"])
        np.testing.assert_array_equal(design["pitch"][..., 0], joint[..., 0])
        np.testing.assert_array_equal(design["time"][..., 1], joint[..., 1])
        np.testing.assert_array_equal(
            design["pitch"][..., 1],
            np.broadcast_to(target[:, 1], (CANDIDATES_PER_TARGET, len(target))),
        )
        np.testing.assert_array_equal(
            design["time"][..., 0],
            np.broadcast_to(target[:, 0], (CANDIDATES_PER_TARGET, len(target))),
        )


def test_every_reported_column_has_the_same_fixed_sample_count():
    metadata = design_metadata()
    assert len(COLUMNS) == 11
    assert metadata["pairs_per_column"] == 8_192
    assert metadata["pairs_per_loss"] == 90_112
