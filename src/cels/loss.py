"""Cumulative Energy Losses on nonnegative frequency--time representations."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import torch
from torch import Tensor, nn

Reduction = Literal["mean", "none"]


class Direction(str, Enum):
    """Accumulation direction on a frequency--time plane."""

    RIGHT_UP = "right_up"
    RIGHT_DOWN = "right_down"
    LEFT_UP = "left_up"
    LEFT_DOWN = "left_down"
    UP = "up"
    DOWN = "down"
    RIGHT = "right"
    LEFT = "left"


DIAGONAL_DIRECTIONS = (
    Direction.RIGHT_UP,
    Direction.RIGHT_DOWN,
    Direction.LEFT_UP,
    Direction.LEFT_DOWN,
)
ORTHOGONAL_DIRECTIONS = (
    Direction.UP,
    Direction.DOWN,
    Direction.RIGHT,
    Direction.LEFT,
)
ALL_DIRECTIONS = DIAGONAL_DIRECTIONS + ORTHOGONAL_DIRECTIONS


@dataclass(frozen=True)
class _DirectionSpec:
    frequency_reverse: bool | None
    time_reverse: bool | None


_SPECS = {
    Direction.RIGHT_UP: _DirectionSpec(False, False),
    Direction.RIGHT_DOWN: _DirectionSpec(True, False),
    Direction.LEFT_UP: _DirectionSpec(False, True),
    Direction.LEFT_DOWN: _DirectionSpec(True, True),
    Direction.UP: _DirectionSpec(False, None),
    Direction.DOWN: _DirectionSpec(True, None),
    Direction.RIGHT: _DirectionSpec(None, False),
    Direction.LEFT: _DirectionSpec(None, True),
}


@dataclass(frozen=True)
class _Geometry:
    frequency_forward: Tensor | None
    frequency_reverse: Tensor | None
    time_forward: Tensor | None
    time_reverse: Tensor | None
    sqrt_weights: Tensor | None


def _directions(values: Iterable[Direction | str]) -> tuple[Direction, ...]:
    result = tuple(Direction(value) for value in values)
    if not result:
        raise ValueError("directions must contain at least one direction")
    if len(set(result)) != len(result):
        raise ValueError("directions must be distinct")
    return result


def _validate_energy(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or value.ndim < 2:
        raise ValueError(f"{name} must have shape [..., frequency, time]")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    detached = value.detach()
    if not bool(torch.isfinite(detached).all()):
        raise ValueError(f"{name} must be finite")
    if bool((detached < 0).any()):
        raise ValueError(f"{name} must be nonnegative")


def _coordinate(
    value: Tensor | None,
    *,
    length: int,
    reference: Tensor,
    name: str,
    required: bool,
) -> Tensor | None:
    if value is None:
        if required:
            raise ValueError(f"{name} coordinates are required by this configuration")
        return None
    result = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if result.ndim != 1 or result.numel() != length:
        raise ValueError(f"{name} coordinates must contain {length} values")
    if not bool(torch.isfinite(result.detach()).all()) or not bool((torch.diff(result) > 0).all()):
        raise ValueError(f"{name} coordinates must be finite and strictly increasing")
    return result


def _reverse_cumsum(value: Tensor, dim: int) -> Tensor:
    return torch.flip(torch.cumsum(torch.flip(value, (dim,)), dim=dim), (dim,))


def _decay_kernel(coordinate: Tensor, horizon: float, reverse: bool) -> Tensor:
    output = coordinate[:, None]
    source = coordinate[None, :]
    signed = output - source
    allowed = signed <= 0 if reverse else signed >= 0
    distance = signed.abs()
    weight = 1.0 - torch.log1p(9.0 * distance / horizon) / math.log(10.0)
    return torch.where(allowed, weight.clamp_min(0.0), torch.zeros_like(weight))


def _widths(coordinate: Tensor, reverse: bool) -> Tensor:
    gap = torch.diff(coordinate)
    zero = coordinate.new_zeros(1)
    return torch.cat((zero, gap) if reverse else (gap, zero))


def _axis_surface(value: Tensor, axis: Literal["frequency", "time"], reverse: bool,
                  kernel: Tensor | None) -> Tensor:
    if kernel is None:
        dim = -2 if axis == "frequency" else -1
        return _reverse_cumsum(value, dim) if reverse else value.cumsum(dim)
    if axis == "frequency":
        return torch.einsum("oi,...it->...ot", kernel, value)
    return torch.einsum("...fi,oi->...fo", value, kernel)


def _surface(value: Tensor, spec: _DirectionSpec, geometry: _Geometry) -> Tensor:
    result = value
    if spec.time_reverse is not None:
        kernel = geometry.time_reverse if spec.time_reverse else geometry.time_forward
        result = _axis_surface(result, "time", spec.time_reverse, kernel)
    if spec.frequency_reverse is not None:
        kernel = (
            geometry.frequency_reverse
            if spec.frequency_reverse
            else geometry.frequency_forward
        )
        result = _axis_surface(result, "frequency", spec.frequency_reverse, kernel)
    return result


class CumulativeEnergyLoss(nn.Module):
    """Compare cumulative views of nonnegative frequency--time energy maps.

    The final two dimensions of each input are interpreted as frequency and
    time. A single target can be broadcast over a candidate batch. Physical
    coordinates are required only for log weighting or finite decay horizons.
    """

    def __init__(
        self,
        *,
        directions: Iterable[Direction | str] = DIAGONAL_DIRECTIONS,
        log_weighting: bool = False,
        time_decay_seconds: float | None = None,
        frequency_decay_hz: float | None = None,
        frequency_floor_hz: float = 20.0,
        eps: float = 1e-12,
        reduction: Reduction = "mean",
    ) -> None:
        super().__init__()
        self.directions = _directions(directions)
        self.log_weighting = bool(log_weighting)
        self.time_decay_seconds = time_decay_seconds
        self.frequency_decay_hz = frequency_decay_hz
        self.frequency_floor_hz = float(frequency_floor_hz)
        self.eps = float(eps)
        self.reduction = reduction
        if reduction not in {"mean", "none"}:
            raise ValueError("reduction must be 'mean' or 'none'")
        for value, name in (
            (time_decay_seconds, "time_decay_seconds"),
            (frequency_decay_hz, "frequency_decay_hz"),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.frequency_floor_hz) or self.frequency_floor_hz <= 0:
            raise ValueError("frequency_floor_hz must be finite and positive")
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be finite and positive")

    def _geometry(
        self,
        reference: Tensor,
        frequency: Tensor | None,
        time: Tensor | None,
    ) -> _Geometry:
        specs = tuple(_SPECS[direction] for direction in self.directions)
        needs_frequency_decay = self.frequency_decay_hz is not None and any(
            spec.frequency_reverse is not None for spec in specs
        )
        needs_time_decay = self.time_decay_seconds is not None and any(
            spec.time_reverse is not None for spec in specs
        )
        frequency = _coordinate(
            frequency,
            length=reference.shape[-2],
            reference=reference,
            name="frequency",
            required=self.log_weighting or needs_frequency_decay,
        )
        time = _coordinate(
            time,
            length=reference.shape[-1],
            reference=reference,
            name="time",
            required=self.log_weighting or needs_time_decay,
        )

        ff = fr = tf = tr = None
        if needs_frequency_decay:
            assert frequency is not None and self.frequency_decay_hz is not None
            ff = _decay_kernel(frequency, self.frequency_decay_hz, False)
            fr = _decay_kernel(frequency, self.frequency_decay_hz, True)
        if needs_time_decay:
            assert time is not None and self.time_decay_seconds is not None
            tf = _decay_kernel(time, self.time_decay_seconds, False)
            tr = _decay_kernel(time, self.time_decay_seconds, True)

        sqrt_weights = None
        if self.log_weighting:
            assert frequency is not None and time is not None
            log_frequency = torch.log2(
                frequency.clamp_min(self.frequency_floor_hz) / self.frequency_floor_hz
            )
            rows = []
            for spec in specs:
                if spec.frequency_reverse is None:
                    fw = (_widths(log_frequency, False) + _widths(log_frequency, True)) / 2
                else:
                    fw = _widths(log_frequency, spec.frequency_reverse)
                if spec.time_reverse is None:
                    tw = (_widths(time, False) + _widths(time, True)) / 2
                else:
                    tw = _widths(time, spec.time_reverse)
                weight = fw[:, None] * tw[None, :]
                total = weight.sum()
                if not bool(torch.isfinite(total.detach())) or not bool(total > 0):
                    raise ValueError("log-weighted representation has zero grid area")
                rows.append(torch.sqrt(weight / total))
            sqrt_weights = torch.stack(rows)
        return _Geometry(ff, fr, tf, tr, sqrt_weights)

    def _prepare(self, target: Tensor, geometry: _Geometry) -> tuple[Tensor, Tensor]:
        mass = target.sum(dim=(-2, -1), keepdim=True)
        if bool((mass.detach() <= 0).any()):
            raise ValueError("every target must contain positive total energy")
        surfaces = torch.stack(
            [_surface(target, _SPECS[direction], geometry) for direction in self.directions],
            dim=-3,
        )
        reference = torch.sqrt((surfaces / mass.unsqueeze(-3)).clamp_min(self.eps))
        return reference.detach(), mass.detach()

    def _terms(
        self,
        candidate: Tensor,
        reference: Tensor,
        mass: Tensor,
        geometry: _Geometry,
    ) -> Tensor:
        surfaces = torch.stack(
            [_surface(candidate, _SPECS[direction], geometry) for direction in self.directions],
            dim=-3,
        )
        feature = torch.sqrt((surfaces / mass.unsqueeze(-3)).clamp_min(self.eps))
        error = feature - reference
        if geometry.sqrt_weights is None:
            return torch.linalg.vector_norm(error.flatten(start_dim=-2), dim=-1) / math.sqrt(
                error.shape[-2] * error.shape[-1]
            )
        return torch.linalg.vector_norm(
            (error * geometry.sqrt_weights).flatten(start_dim=-2), dim=-1
        )

    def directional_distances(
        self,
        candidate: Tensor,
        target: Tensor,
        *,
        frequency: Tensor | None = None,
        time: Tensor | None = None,
    ) -> Tensor:
        _validate_energy(candidate, "candidate")
        _validate_energy(target, "target")
        candidate, target = torch.broadcast_tensors(candidate, target)
        geometry = self._geometry(target, frequency, time)
        reference, mass = self._prepare(target, geometry)
        return self._terms(candidate, reference, mass, geometry)

    def forward(
        self,
        candidate: Tensor,
        target: Tensor,
        *,
        frequency: Tensor | None = None,
        time: Tensor | None = None,
    ) -> Tensor:
        value = self.directional_distances(
            candidate, target, frequency=frequency, time=time
        ).mean(dim=-1)
        return value.mean() if self.reduction == "mean" else value

    def bind(
        self,
        target: Tensor,
        *,
        frequency: Tensor | None = None,
        time: Tensor | None = None,
    ) -> BoundCumulativeEnergyLoss:
        """Cache target surfaces for repeated candidate comparisons."""
        _validate_energy(target, "target")
        target = target.detach()
        geometry = self._geometry(target, frequency, time)
        reference, mass = self._prepare(target, geometry)
        return BoundCumulativeEnergyLoss(self, reference, mass, geometry)


class BoundCumulativeEnergyLoss(nn.Module):
    """A Cumulative Energy Loss with cached target features."""

    def __init__(
        self,
        source: CumulativeEnergyLoss,
        reference: Tensor,
        mass: Tensor,
        geometry: _Geometry,
    ) -> None:
        super().__init__()
        self.directions = source.directions
        self.eps = source.eps
        self.reduction = source.reduction
        self.register_buffer("reference", reference)
        self.register_buffer("mass", mass)
        for name, value in (
            ("frequency_forward", geometry.frequency_forward),
            ("frequency_reverse", geometry.frequency_reverse),
            ("time_forward", geometry.time_forward),
            ("time_reverse", geometry.time_reverse),
            ("sqrt_weights", geometry.sqrt_weights),
        ):
            self.register_buffer(name, value if value is not None else reference.new_empty(0))

    @staticmethod
    def _optional(value: Tensor) -> Tensor | None:
        return None if value.numel() == 0 else value

    def _geometry(self) -> _Geometry:
        return _Geometry(
            self._optional(self.frequency_forward),
            self._optional(self.frequency_reverse),
            self._optional(self.time_forward),
            self._optional(self.time_reverse),
            self._optional(self.sqrt_weights),
        )

    def directional_distances(self, candidate: Tensor) -> Tensor:
        _validate_energy(candidate, "candidate")
        if candidate.shape[-2:] != self.reference.shape[-2:]:
            raise ValueError("candidate and bound target grids differ")
        geometry = self._geometry()
        surfaces = torch.stack(
            [
                _surface(candidate, _SPECS[direction], geometry)
                for direction in self.directions
            ],
            dim=-3,
        )
        feature = torch.sqrt((surfaces / self.mass.unsqueeze(-3)).clamp_min(self.eps))
        error = feature - self.reference
        weights = self._optional(self.sqrt_weights)
        if weights is None:
            return torch.linalg.vector_norm(error.flatten(start_dim=-2), dim=-1) / math.sqrt(
                error.shape[-2] * error.shape[-1]
            )
        return torch.linalg.vector_norm((error * weights).flatten(start_dim=-2), dim=-1)

    def forward(self, candidate: Tensor) -> Tensor:
        value = self.directional_distances(candidate).mean(dim=-1)
        return value.mean() if self.reduction == "mean" else value


__all__ = [
    "ALL_DIRECTIONS",
    "BoundCumulativeEnergyLoss",
    "CumulativeEnergyLoss",
    "DIAGONAL_DIRECTIONS",
    "Direction",
    "ORTHOGONAL_DIRECTIONS",
]
