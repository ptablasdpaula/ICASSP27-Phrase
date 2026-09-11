"""Internal differentiable fractional-delay primitives."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def lagrange_weights(evaluation: Tensor, order: int) -> Tensor:
    """Return order-``order`` maximally-flat Lagrange FIR weights."""
    if order < 1:
        raise ValueError("Lagrange order must be positive")
    nodes = torch.arange(order + 1, dtype=evaluation.dtype, device=evaluation.device)
    rows = []
    for index in range(order + 1):
        others = torch.cat((nodes[:index], nodes[index + 1 :]))
        rows.append(
            torch.prod(
                (evaluation[..., None] - others) / (nodes[index] - others), dim=-1
            )
        )
    return torch.stack(rows, dim=-1)


def lagrange_anchor(delay_samples: Tensor, order: int) -> Tensor:
    """Choose the paper's causal, centred-when-possible FIR anchor."""
    delay = delay_samples.reshape(-1)
    offset = torch.where(
        delay.detach() >= float(order) / 2.0,
        delay.new_tensor(float(order // 2)),
        delay.new_zeros(()),
    )
    whole = torch.floor(delay - offset).detach().to(torch.long)
    if bool((whole < 0).any()):
        raise ValueError("Lagrange delay would require a noncausal advance")
    return whole


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


def fft_convolve_rows(left: Tensor, right: Tensor, length: int) -> Tensor:
    """Row-separable differentiable linear convolution."""
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("convolution operands must be matched matrices")
    if length < 1:
        raise ValueError("convolution length must be positive")
    full = left.shape[-1] + right.shape[-1] - 1
    fft_length = 1 << max(max(full, length) - 1, 0).bit_length()
    return torch.fft.irfft(
        torch.fft.rfft(left, n=fft_length, dim=-1)
        * torch.fft.rfft(right, n=fft_length, dim=-1),
        n=fft_length,
        dim=-1,
    )[..., :length]


def power_series_inverse_rows(denominator: Tensor, length: int) -> Tensor:
    """First ``length`` coefficients of independent ``1/A(z)`` series."""
    if denominator.ndim != 2 or length < 1:
        raise ValueError("invalid denominator dimensions")
    if bool((denominator[:, 0].detach().abs() < 1e-14).any()):
        raise FloatingPointError("IIR denominator has a zero leading term")
    inverse = 1.0 / denominator[:, :1]
    while inverse.shape[-1] < length:
        width = min(2 * inverse.shape[-1], length)
        product = fft_convolve_rows(denominator[:, :width], inverse, width)
        residual = -product
        residual = residual.index_add(
            1,
            torch.zeros(1, dtype=torch.long, device=residual.device),
            residual.new_full((residual.shape[0], 1), 2.0),
        )
        inverse = fft_convolve_rows(inverse, residual, width)
    return inverse


def zero_state_filter_rows(source: Tensor, numerator: Tensor, denominator: Tensor) -> Tensor:
    """Evaluate exact finite causal prefixes of rational row filters."""
    if source.ndim != 2 or not (
        source.shape[0] == numerator.shape[0] == denominator.shape[0]
    ):
        raise ValueError("filter operands must have matching row counts")
    count = source.shape[-1]
    forcing = fft_convolve_rows(source, numerator, count)
    inverse = power_series_inverse_rows(denominator, count)
    result = fft_convolve_rows(forcing, inverse, count)
    if not bool(torch.isfinite(result.detach()).all()):
        raise FloatingPointError("fractional-delay filtering produced non-finite audio")
    return result


def causal_relocate_rows(rows: Tensor, whole: Tensor, output_count: int) -> Tensor:
    """Apply detached nonnegative integer shifts to a matrix of signals."""
    if bool((whole < 0).detach().any()):
        raise ValueError("causal relocation cannot contain an advance")
    sample = torch.arange(output_count, device=rows.device)[None, :]
    read = sample - whole.detach().to(torch.long)[:, None]
    valid = (read >= 0) & (read < rows.shape[-1])
    return rows.gather(1, read.clamp(0, rows.shape[-1] - 1)) * valid


__all__ = [
    "causal_relocate_rows",
    "lagrange_anchor",
    "lagrange_weights",
    "thiran_anchor",
    "thiran_denominator",
    "zero_state_filter_rows",
]
