"""Internal differentiable fractional-delay primitives."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def thiran_anchor(delay_samples: Tensor, order: int) -> Tensor:
    """Choose the fixed causal anchor used by the Thiran sections."""
    delay = delay_samples.reshape(-1)
    return torch.floor(delay - float(order) + 0.5).detach().to(torch.long).clamp_min(0)


def thiran_denominator(delay_samples: Tensor, order: int, whole: Tensor) -> Tensor:
    """Return differentiable standard Thiran denominator coefficients."""
    delay = delay_samples.reshape(-1)
    anchor = whole.reshape(-1)
    if delay.shape != anchor.shape or order < 1:
        raise ValueError("invalid Thiran delay, order, or anchor")
    if not bool(torch.isfinite(delay.detach()).all()):
        raise ValueError("Thiran delay must be finite")
    evaluation = delay - anchor.to(delay.dtype)
    if bool((evaluation <= float(order - 1)).detach().any()):
        raise ValueError("Thiran all-pass lies outside its stable delay range")
    taps = torch.arange(order + 1, dtype=delay.dtype, device=delay.device)
    base = evaluation[:, None] - float(order) + taps[None, :]
    coefficients = [torch.ones_like(evaluation)]
    for index in range(1, order + 1):
        product = torch.prod(base / (base + float(index)), dim=-1)
        coefficients.append(((-1.0) ** index) * math.comb(order, index) * product)
    return torch.stack(coefficients, dim=-1)


def hermitian_rfft_projection(spectrum: Tensor, nfft: int) -> Tensor:
    """Make the self-conjugate bins of an even-length real FFT explicitly real."""
    if isinstance(nfft, bool) or not isinstance(nfft, int) or nfft < 2 or nfft % 2:
        raise ValueError("nfft must be a positive even integer")
    if not spectrum.is_complex() or spectrum.shape[-1] != nfft // 2 + 1:
        raise ValueError("spectrum is not a compatible one-sided real FFT")
    zero = torch.zeros_like(spectrum[..., :1].real)
    projected = torch.cat(
        (
            torch.complex(spectrum[..., :1].real, zero),
            spectrum[..., 1:-1],
            torch.complex(spectrum[..., -1:].real, zero),
        ),
        dim=-1,
    )
    if not bool(torch.isfinite(projected.detach()).all()):
        raise FloatingPointError("Hermitian projection produced non-finite values")
    return projected
