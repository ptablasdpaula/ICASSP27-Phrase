"""Evaluation-only metrics for the paper experiments.

Nothing in this module contributes to an optimisation objective or an audio
gradient. Coordinates are expressed as ``(log2(f0 / 80 Hz), onset seconds)``.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor


def hungarian_assignment(candidate: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, bool]:
    """Match events with one octave assigned the same cost as one second."""
    candidate = np.asarray(candidate, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if candidate.shape != target.shape or candidate.ndim != 2 or candidate.shape[-1] != 2:
        raise ValueError("candidate and target must have matching [event,2] shapes")
    cost = np.square(candidate[:, None] - target[None]).sum(-1)
    rows, columns = linear_sum_assignment(cost)
    best = cost[rows, columns].sum()
    tied = False
    if len(candidate) > 1:
        for row, column in zip(rows, columns, strict=True):
            alternative = cost.copy()
            alternative[row, column] = np.inf
            alt_rows, alt_columns = linear_sum_assignment(alternative)
            if alternative[alt_rows, alt_columns].sum() - best <= 1e-12:
                tied = True
                break
    return columns, tied


def phrase_gradient_cosine(
    gradients: np.ndarray,
    candidates: np.ndarray,
    target: np.ndarray,
    assignments: np.ndarray,
    ties: np.ndarray,
) -> np.ndarray:
    """Cosine between whole-phrase descent and matched target displacement."""
    gradients = np.asarray(gradients)
    displacement = (target[assignments] - candidates)[:, None]
    descent = -gradients
    dot = (descent * displacement).sum(axis=(-2, -1))
    norms = np.linalg.norm(descent, axis=(-2, -1)) * np.linalg.norm(
        displacement, axis=(-2, -1)
    )
    cosine = np.divide(dot, norms, out=np.zeros_like(dot), where=norms > 0)
    cosine = np.clip(cosine, -1.0, 1.0)
    active = (np.abs(displacement[:, 0]) > 1e-12).any(axis=(-2, -1))
    cosine[~(~ties & active)] = np.nan
    return cosine


def recovery_metrics(
    f0_hz: np.ndarray,
    onset_seconds: np.ndarray,
    target_f0_hz: np.ndarray,
    target_onset_seconds: np.ndarray,
) -> dict[str, object]:
    """Hungarian-assigned pitch and onset errors for one recovered phrase."""
    pitch = np.log2(np.asarray(f0_hz) / 80.0)
    target_pitch = np.log2(np.asarray(target_f0_hz) / 80.0)
    onset = np.asarray(onset_seconds)
    target_onset = np.asarray(target_onset_seconds)
    candidate = np.stack((pitch, onset), axis=-1)
    target = np.stack((target_pitch, target_onset), axis=-1)
    assignment, tied = hungarian_assignment(candidate, target)
    return {
        "pitch_mae_cents": float(np.abs(1200.0 * (pitch - target_pitch[assignment])).mean()),
        "onset_mae_ms": float(np.abs(1000.0 * (onset - target_onset[assignment])).mean()),
        "assignment": assignment.tolist(),
        "assignment_tied": tied,
    }


def log_spectral_distance(
    candidate: Tensor,
    target: Tensor,
    *,
    n_fft: int = 1024,
    hop_length: int = 256,
    floor_db: float = -100.0,
) -> Tensor:
    """RMS log-magnitude error with a common target-relative floor.

    This is an evaluation metric. It is deliberately separate from the
    differentiable objectives in :mod:`icassp27_phrase.losses`.
    """
    if candidate.shape != target.shape or candidate.ndim not in (1, 2):
        raise ValueError("candidate and target must have matching [samples] or [batch,samples]")
    if not candidate.is_floating_point() or candidate.dtype != target.dtype:
        raise TypeError("candidate and target must share a floating-point dtype")
    if n_fft <= 1 or hop_length <= 0 or hop_length > n_fft:
        raise ValueError("n_fft and hop_length must define a valid STFT")
    if floor_db >= 0.0:
        raise ValueError("floor_db must be negative")
    squeeze = candidate.ndim == 1
    candidate_rows = candidate[None] if squeeze else candidate
    target_rows = target[None] if squeeze else target
    window = torch.hann_window(
        n_fft, periodic=True, dtype=candidate.dtype, device=candidate.device
    )

    def magnitude(audio: Tensor) -> Tensor:
        return torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            center=True,
            pad_mode="constant",
            onesided=True,
            return_complex=True,
        ).abs()

    target_magnitude = magnitude(target_rows)
    candidate_magnitude = magnitude(candidate_rows)
    ratio = 10.0 ** (floor_db / 20.0)
    floor = (target_magnitude.amax((-2, -1), keepdim=True) * ratio).clamp_min(
        torch.finfo(candidate.dtype).tiny
    )
    target_db = 20.0 * torch.log10(torch.maximum(target_magnitude, floor))
    candidate_db = 20.0 * torch.log10(torch.maximum(candidate_magnitude, floor))
    distance = (candidate_db - target_db).square().mean((-2, -1)).sqrt()
    return distance[0] if squeeze else distance


__all__ = [
    "hungarian_assignment",
    "log_spectral_distance",
    "phrase_gradient_cosine",
    "recovery_metrics",
]
