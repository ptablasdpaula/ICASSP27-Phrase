from __future__ import annotations

import numpy as np
import torch
from icassp27_phrase.metrics import (
    hungarian_assignment,
    log_spectral_distance,
    phrase_gradient_cosine,
    recovery_metrics,
)


def test_hungarian_cost_uses_octaves_and_seconds_equally() -> None:
    target = np.array([[0.0, 0.0], [1.0, 1.0]])
    candidate = np.array([[1.05, 1.0], [0.0, 0.05]])
    assignment, tied = hungarian_assignment(candidate, target)
    np.testing.assert_array_equal(assignment, [1, 0])
    assert not tied


def test_phrase_cosine_uses_the_complete_matched_phrase() -> None:
    target = np.array([[0.0, 0.0], [1.0, 1.0]])
    candidates = np.array([[[0.2, 0.0], [1.0, 0.8]]])
    assignments = np.array([[0, 1]])
    gradients = np.array([[[[1.0, 0.0], [0.0, 1.0]]]])
    cosine = phrase_gradient_cosine(
        gradients, candidates, target, assignments, np.array([False])
    )
    np.testing.assert_allclose(cosine, [[0.0]], atol=1e-15)


def test_recovery_metrics_report_cents_and_milliseconds() -> None:
    result = recovery_metrics(
        np.array([160.0, 80.0]),
        np.array([1.01, 0.22]),
        np.array([80.0, 160.0]),
        np.array([0.2, 1.0]),
    )
    assert result["pitch_mae_cents"] == 0.0
    assert result["onset_mae_ms"] == 15.0
    assert result["assignment"] == [1, 0]


def test_log_spectral_distance_is_zero_only_for_identical_audio() -> None:
    time = torch.arange(4096, dtype=torch.float64) / 16_000
    target = torch.sin(2 * torch.pi * 220 * time)
    shifted = torch.roll(target, 9)
    assert float(log_spectral_distance(target, target)) == 0.0
    assert float(log_spectral_distance(shifted, target)) > 0.0
