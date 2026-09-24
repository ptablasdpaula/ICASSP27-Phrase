"""Differentiable pickup-free two-rail digital waveguide."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..runtime import require_df2_backend
from ._fractional import (
    thiran_anchor,
    thiran_denominator,
)
from .config import F0_BOUNDS_HZ, ONSET_BOUNDS_SECONDS, WaveguideConfig


@dataclass(frozen=True)
class RationalTransfer:
    """One causal rational block in ascending powers of ``z^-1``."""

    numerator: Tensor
    denominator: Tensor


@dataclass(frozen=True)
class WaveguideTransfer:
    """The physical segment designs used by the two travelling rails."""

    long_segment: RationalTransfer
    short_segment: RationalTransfer
    loop_filter: RationalTransfer
    rail_delay_samples: Tensor
    long_delay_samples: Tensor
    short_delay_samples: Tensor


@dataclass(frozen=True)
class _RationalRows:
    numerator: Tensor
    denominator: Tensor


def _batch_poly_pad(value: Tensor, width: int) -> Tensor:
    return F.pad(value, (0, width - value.shape[-1]))


def _batch_poly_add(left: Tensor, right: Tensor) -> Tensor:
    width = max(left.shape[-1], right.shape[-1])
    return _batch_poly_pad(left, width) + _batch_poly_pad(right, width)


def _batch_poly_mul(left: Tensor, right: Tensor) -> Tensor:
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("batched polynomials need matching leading dimensions")
    degree = (
        (
            torch.arange(left.shape[1], device=left.device)[:, None]
            + torch.arange(right.shape[1], device=right.device)[None, :]
        )
        .reshape(1, -1)
        .expand(left.shape[0], -1)
    )
    products = (left[:, :, None] * right[:, None, :]).reshape(left.shape[0], -1)
    return left.new_zeros((left.shape[0], left.shape[1] + right.shape[1] - 1)).scatter_add(
        1, degree, products
    )


def _rational_multiply(left: _RationalRows, right: _RationalRows) -> _RationalRows:
    return _RationalRows(
        _batch_poly_mul(left.numerator, right.numerator),
        _batch_poly_mul(left.denominator, right.denominator),
    )


def _rational_add(
    left: _RationalRows, right: _RationalRows, *, right_scale: float = 1.0
) -> _RationalRows:
    numerator = _batch_poly_add(
        _batch_poly_mul(left.numerator, right.denominator),
        float(right_scale) * _batch_poly_mul(right.numerator, left.denominator),
    )
    return _RationalRows(numerator, _batch_poly_mul(left.denominator, right.denominator))


def _rational_scale(value: _RationalRows, scale: float) -> _RationalRows:
    return _RationalRows(float(scale) * value.numerator, value.denominator)


def _row_response(
    delay: Tensor,
    method: str,
    order: int,
    whole: Tensor,
    omega: Tensor,
) -> Tensor:
    exponent = torch.complex(torch.zeros_like(omega), -omega)[:, None]
    taps = whole[:, None] + torch.arange(order + 1, device=delay.device)[None, :]
    if method == "thiran":
        denominator = thiran_denominator(delay, order, whole)
        denominator_taps = torch.arange(order + 1, dtype=delay.dtype, device=delay.device)[None, :]
        numerator = torch.sum(
            torch.flip(denominator, dims=(-1,)) * torch.exp(exponent * taps.to(omega.dtype)),
            dim=-1,
        )
        denominator_response = torch.sum(
            denominator * torch.exp(exponent * denominator_taps), dim=-1
        )
        return numerator / denominator_response
    raise ValueError(f"{method!r} has no time-domain delay response")


def _unwrapped_phase(
    delay: Tensor,
    method: str,
    order: int,
    whole: Tensor,
    omega: Tensor,
) -> Tensor:
    wrapped = torch.angle(_row_response(delay, method, order, whole, omega))
    ideal = -omega * delay
    turns = torch.round((ideal - wrapped) / (2.0 * math.pi)).detach()
    return wrapped + 2.0 * math.pi * turns


def _delay_rows(delay: Tensor, method: str, order: int, whole: Tensor) -> _RationalRows:
    taps = whole[:, None] + torch.arange(order + 1, device=delay.device)[None, :]
    width = int(taps.max().detach().cpu().item()) + 1
    if method == "thiran":
        denominator = thiran_denominator(delay, order, whole)
        numerator = delay.new_zeros((delay.numel(), width)).scatter_add(
            1, taps, torch.flip(denominator, dims=(-1,))
        )
        return _RationalRows(numerator, denominator)
    raise ValueError(f"{method!r} has no rational delay section")


def _segment_rows(
    f0_hz: Tensor, config: WaveguideConfig
) -> tuple[_RationalRows, _RationalRows, Tensor]:
    """Build phase-corrected long and short physical segment delays."""
    orders = config.segment_orders
    long_order, short_order = orders
    frequency = f0_hz.reshape(-1)
    omega = 2.0 * math.pi * frequency / float(config.sample_rate)
    pole = frequency.new_tensor(config.loop_pole)
    loop_phase = -torch.atan2(pole * torch.sin(omega), 1.0 - pole * torch.cos(omega))
    rail = 0.5 * (float(config.sample_rate) / frequency + loop_phase / omega)
    beta = float(config.pluck_position)
    initial_long = (1.0 - beta) * rail
    initial_short = beta * rail
    long_whole = thiran_anchor(initial_long, long_order)
    short_whole = thiran_anchor(initial_short, short_order)
    for _ in range(config.phase_correction_iterations):
        long_phase = _unwrapped_phase(
            (1.0 - beta) * rail,
            config.interpolation,
            long_order,
            long_whole,
            omega,
        )
        short_phase = _unwrapped_phase(
            beta * rail,
            config.interpolation,
            short_order,
            short_whole,
            omega,
        )
        residual = 2.0 * (long_phase + short_phase) + loop_phase + 2.0 * math.pi
        rail = rail + residual / (2.0 * omega)
    long = _delay_rows(
        (1.0 - beta) * rail,
        config.interpolation,
        long_order,
        long_whole,
    )
    short = _delay_rows(
        beta * rail,
        config.interpolation,
        short_order,
        short_whole,
    )
    if bool((long.numerator[:, 0].detach() != 0.0).any()):
        raise ValueError("the long segment unexpectedly has feedthrough")
    return long, short, rail


def _loop_rows(f0_hz: Tensor, config: WaveguideConfig) -> _RationalRows:
    frequency = f0_hz.reshape(-1)
    return _RationalRows(
        torch.full_like(frequency[:, None], config.loop_gain * (1.0 - config.loop_pole)),
        torch.stack(
            (torch.ones_like(frequency), torch.full_like(frequency, -config.loop_pole)),
            dim=1,
        ),
    )


def _collapsed_rows(
    f0_hz: Tensor, config: WaveguideConfig
) -> tuple[_RationalRows, _RationalRows, _RationalRows, Tensor]:
    """Collapse the exact pickup-free two-rail graph into one rational filter."""
    long, short, rail_delay = _segment_rows(f0_hz, config)
    loop = _loop_rows(f0_hz, config)
    rail = _rational_multiply(long, short)
    feedback = _rational_multiply(loop, _rational_multiply(rail, rail))
    identity = _RationalRows(
        torch.ones_like(f0_hz.reshape(-1, 1)),
        torch.ones_like(f0_hz.reshape(-1, 1)),
    )
    characteristic = _rational_add(identity, feedback, right_scale=-1.0)
    closed_loop = _RationalRows(characteristic.denominator, characteristic.numerator)
    nut_comb = _rational_add(identity, _rational_multiply(long, long), right_scale=-1.0)
    source_to_bridge = _rational_scale(_rational_multiply(short, nut_comb), 0.5)
    transfer = _rational_multiply(source_to_bridge, closed_loop)
    return transfer, long, short, rail_delay


def _normalized_transfer_rows(f0_hz: Tensor, config: WaveguideConfig) -> tuple[Tensor, Tensor]:
    transfer, _, _, _ = _collapsed_rows(f0_hz, config)
    leading = transfer.denominator[:, :1]
    if not bool(torch.isfinite(leading.detach()).all()) or bool(
        (leading.detach().abs() < 1e-14).any()
    ):
        raise FloatingPointError("waveguide transfer has a singular leading term")
    return transfer.numerator / leading, transfer.denominator[:, 1:] / leading


def _df2_filter(source: Tensor, numerator: Tensor, denominator: Tensor) -> Tensor:
    """Evaluate constant rational rows through PhilTorch/TorchLPC DF2."""
    if source.ndim != 2 or numerator.ndim != 2 or denominator.ndim != 2:
        raise ValueError("DF2 operands must be batched matrices")
    if not (source.shape[0] == numerator.shape[0] == denominator.shape[0]):
        raise ValueError("DF2 operands must have matching row counts")
    if not (source.device == numerator.device == denominator.device):
        raise ValueError("DF2 operands must share a device")
    if denominator.shape[-1] < 2:
        raise RuntimeError("the registered DF2 path requires the TorchLPC LPC recurrence")
    require_df2_backend(source.device)
    try:
        from philtorch.lpv import lfilter
    except ImportError as error:  # pragma: no cover - installation-specific
        raise RuntimeError("install the pinned PhilTorch and TorchLPC backends") from error
    samples = source.shape[1]
    b_trace = numerator[:, None, :].expand(-1, samples, -1)
    a_trace = denominator[:, None, :].expand(-1, samples, -1)
    return lfilter(
        b_trace.contiguous(),
        a_trace.contiguous(),
        source.contiguous(),
        form="df2",
        backend="torchlpc",
    )


def _regimes(boundary_row: list[int]) -> list[tuple[int, int]]:
    """Collapse coincident boundaries to the last active event."""
    result = [(0, 0)]
    for event, boundary in enumerate(boundary_row[1:], start=1):
        if boundary == result[-1][0]:
            result[-1] = (boundary, event)
        else:
            result.append((boundary, event))
    return result


def _hard_reset_df2(
    source: Tensor, numerator: Tensor, denominator: Tensor, boundaries: Tensor
) -> Tensor:
    if numerator.shape[1] == 1:
        return _df2_filter(source, numerator[:, 0], denominator[:, 0])
    regime_rows = [_regimes(row) for row in boundaries.detach().cpu().tolist()]
    pieces: list[list[Tensor]] = [[] for _ in range(source.shape[0])]
    for regime_index in range(max(map(len, regime_rows))):
        rows: list[int] = []
        events: list[int] = []
        starts: list[int] = []
        stops: list[int] = []
        for batch_index, regimes in enumerate(regime_rows):
            if regime_index >= len(regimes):
                continue
            start, event = regimes[regime_index]
            stop = (
                regimes[regime_index + 1][0] if regime_index + 1 < len(regimes) else source.shape[1]
            )
            if stop > start:
                rows.append(batch_index)
                events.append(event)
                starts.append(start)
                stops.append(stop)
        if not rows:
            continue
        lengths = [stop - start for start, stop in zip(starts, stops, strict=True)]
        width = max(lengths)
        segments = torch.stack(
            [
                F.pad(source[row, start:stop], (0, width - length))
                for row, start, stop, length in zip(rows, starts, stops, lengths, strict=True)
            ]
        )
        filtered = _df2_filter(
            segments,
            torch.stack([numerator[row, event] for row, event in zip(rows, events, strict=True)]),
            torch.stack([denominator[row, event] for row, event in zip(rows, events, strict=True)]),
        )
        for local, (row, length) in enumerate(zip(rows, lengths, strict=True)):
            pieces[row].append(filtered[local, :length])
    return torch.stack([torch.cat(row) for row in pieces])


def _persistent_df2(
    source: Tensor, numerator: Tensor, denominator: Tensor, boundaries: Tensor
) -> Tensor:
    require_df2_backend(source.device)
    sample = torch.arange(source.shape[1], device=source.device)
    selected = (
        torch.sum(boundaries[:, :, None] <= sample[None, None, :], dim=1).clamp(
            1, numerator.shape[1]
        )
        - 1
    )
    batch = torch.arange(source.shape[0], device=source.device)[:, None]
    if denominator.shape[-1] < 2:
        raise RuntimeError("the registered DF2 path requires the TorchLPC LPC recurrence")
    try:
        from philtorch.lpv import lfilter
    except ImportError as error:  # pragma: no cover - installation-specific
        raise RuntimeError("install the pinned PhilTorch and TorchLPC backends") from error
    return lfilter(
        numerator[batch, selected].contiguous(),
        denominator[batch, selected].contiguous(),
        source.contiguous(),
        form="df2",
        backend="torchlpc",
    )


class Waveguide(nn.Module):
    """Two-rail Thiran waveguide using the compiled DF2 recurrence.

    State resets at event boundaries or persists for the State diagnostic.
    """

    def __init__(self, config: WaveguideConfig | None = None) -> None:
        super().__init__()
        self.config = config or WaveguideConfig()

    def transfer(self, f0_hz: Tensor | float) -> WaveguideTransfer:
        """Return the physical time-domain segment designs for one frequency."""
        frequency = torch.as_tensor(f0_hz).reshape(())
        if not frequency.is_floating_point():
            frequency = frequency.to(torch.float64)
        if frequency.dtype != torch.float64:
            raise TypeError("waveguide controls use torch.float64")
        if not bool(torch.isfinite(frequency.detach())) or not bool(
            (frequency.detach() >= F0_BOUNDS_HZ[0]) & (frequency.detach() <= F0_BOUNDS_HZ[1])
        ):
            raise ValueError("f0_hz lies outside 80--320 Hz")
        _, long, short, rail = _collapsed_rows(frequency[None], self.config)
        loop = _loop_rows(frequency[None], self.config)
        beta = self.config.pluck_position
        return WaveguideTransfer(
            long_segment=RationalTransfer(long.numerator[0], long.denominator[0]),
            short_segment=RationalTransfer(short.numerator[0], short.denominator[0]),
            loop_filter=RationalTransfer(loop.numerator[0], loop.denominator[0]),
            rail_delay_samples=rail[0],
            long_delay_samples=(1.0 - beta) * rail[0],
            short_delay_samples=beta * rail[0],
        )

    def forward(self, source: Tensor, f0_hz: Tensor, onset_seconds: Tensor) -> Tensor:
        """Filter a summed source using controls shaped ``[batch,event]``."""
        if source.ndim != 2 or source.shape[1] != self.config.sample_count:
            raise ValueError("source must have shape [batch,sample_count]")
        if f0_hz.ndim != 2 or onset_seconds.shape != f0_hz.shape:
            raise ValueError("controls must have matching [batch,event] shapes")
        if source.shape[0] != f0_hz.shape[0] or f0_hz.shape[1] < 1:
            raise ValueError("source and non-empty controls need matching batches")
        if (
            source.dtype != torch.float64
            or f0_hz.dtype != torch.float64
            or onset_seconds.dtype != torch.float64
        ):
            raise TypeError("the waveguide is float64 throughout")
        if not (source.device == f0_hz.device == onset_seconds.device):
            raise ValueError("source and controls must share a device")
        if not all(
            bool(torch.isfinite(value.detach()).all()) for value in (source, f0_hz, onset_seconds)
        ):
            raise FloatingPointError("source and controls must be finite")
        if not bool(
            ((f0_hz.detach() >= F0_BOUNDS_HZ[0]) & (f0_hz.detach() <= F0_BOUNDS_HZ[1])).all()
        ):
            raise ValueError("f0_hz lies outside 80--320 Hz")
        if not bool(
            (
                (onset_seconds.detach() >= ONSET_BOUNDS_SECONDS[0])
                & (onset_seconds.detach() <= ONSET_BOUNDS_SECONDS[1])
            ).all()
        ):
            raise ValueError("onset_seconds lies outside 0.2--1.8 s")

        order = torch.argsort(onset_seconds.detach(), dim=1, stable=True)
        ordered_f0 = torch.gather(f0_hz, 1, order)
        ordered_onset = torch.gather(onset_seconds, 1, order)
        boundaries = torch.floor(ordered_onset.detach() * float(self.config.sample_rate)).to(
            torch.long
        )

        flat = ordered_f0.reshape(-1)
        event_shape = tuple(ordered_f0.shape)
        numerator_rows, denominator_rows = _normalized_transfer_rows(flat, self.config)
        numerator = numerator_rows.reshape(*event_shape, -1)
        denominator = denominator_rows.reshape(*event_shape, -1)
        audio = (
            _hard_reset_df2(source, numerator, denominator, boundaries)
            if self.config.state_policy == "hard_reset"
            else _persistent_df2(source, numerator, denominator, boundaries)
        )
        if audio.shape != source.shape or audio.dtype != torch.float64:
            raise RuntimeError("waveguide returned malformed audio")
        if not bool(torch.isfinite(audio.detach()).all()):
            raise FloatingPointError("waveguide produced non-finite audio")
        return audio


__all__ = ["RationalTransfer", "Waveguide", "WaveguideTransfer"]
