"""Typed configuration for the renderer and the registered optimiser."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

F0_BOUNDS_HZ = (80.0, 320.0)
ONSET_BOUNDS_SECONDS = (0.2, 1.8)
CARDINALITIES = (1, 2, 4, 6, 8)
SAMPLE_RATE = 4_000
SAMPLE_COUNT = 8_000
MASTER_SEED = 2027

OnsetMethod = Literal["naive", "lagrange", "fourier", "thiran"]
InterpolationMethod = Literal["linear", "lagrange", "fourier", "thiran"]
Realization = Literal["df2", "literal", "frequency"]
StatePolicy = Literal["hard_reset", "persistent"]


@dataclass(frozen=True)
class EventPhrase:
    """A known-cardinality phrase represented by paired float64 vectors."""

    f0_hz: Tensor
    onset_seconds: Tensor

    def __post_init__(self) -> None:
        if self.f0_hz.ndim != 1 or self.onset_seconds.shape != self.f0_hz.shape:
            raise ValueError("f0_hz and onset_seconds must be matching vectors")
        if self.f0_hz.numel() < 1:
            raise ValueError("a phrase must contain at least one event")
        if self.f0_hz.dtype != torch.float64 or self.onset_seconds.dtype != torch.float64:
            raise TypeError("phrase controls must use torch.float64")
        if self.f0_hz.device != self.onset_seconds.device:
            raise ValueError("phrase controls must share a device")
        if not bool(torch.isfinite(self.f0_hz.detach()).all()) or not bool(
            torch.isfinite(self.onset_seconds.detach()).all()
        ):
            raise ValueError("phrase controls must be finite")
        if not bool(
            (
                (self.f0_hz.detach() >= F0_BOUNDS_HZ[0])
                & (self.f0_hz.detach() <= F0_BOUNDS_HZ[1])
            ).all()
        ):
            raise ValueError("f0_hz lies outside 80--320 Hz")
        if not bool(
            (
                (self.onset_seconds.detach() >= ONSET_BOUNDS_SECONDS[0])
                & (self.onset_seconds.detach() <= ONSET_BOUNDS_SECONDS[1])
            ).all()
        ):
            raise ValueError("onset_seconds lies outside 0.2--1.8 s")

    @property
    def cardinality(self) -> int:
        return int(self.f0_hz.numel())


@dataclass(frozen=True)
class ExciterConfig:
    """Half-raised-cosine source used in the paper."""

    method: OnsetMethod = "fourier"
    sample_rate: int = SAMPLE_RATE
    sample_count: int = SAMPLE_COUNT
    amplitude: float = 0.8
    duration_seconds: float = 0.010
    lagrange_order: int = 5
    thiran_order: int = 1
    fourier_fft_length: int = 16_384

    def __post_init__(self) -> None:
        if self.method not in ("naive", "lagrange", "fourier", "thiran"):
            raise ValueError(f"unknown onset method {self.method!r}")
        if self.sample_rate != SAMPLE_RATE or self.sample_count < 1:
            raise ValueError("the registered renderer is fixed at 4 kHz")
        if not (self.amplitude > 0.0 and self.duration_seconds > 0.0):
            raise ValueError("exciter amplitude and duration must be positive")
        if self.lagrange_order != 5 or self.thiran_order != 1:
            raise ValueError("registered onset orders are Lagrange-5 and Thiran-1")
        if self.method == "fourier" and (
            self.fourier_fft_length % 2
            or self.fourier_fft_length < 2 * self.sample_count
        ):
            raise ValueError(
                "Fourier onset placement requires an even FFT at least twice "
                "the rendered impulse-response horizon"
            )


@dataclass(frozen=True)
class WaveguideConfig:
    """Pickup-free two-rail digital waveguide configuration."""

    interpolation: InterpolationMethod = "thiran"
    realization: Realization = "df2"
    state_policy: StatePolicy = "hard_reset"
    sample_rate: int = SAMPLE_RATE
    sample_count: int = SAMPLE_COUNT
    loop_gain: float = 0.99
    loop_pole: float = 0.2
    pluck_position: float = 0.23
    phase_correction_iterations: int = 12
    fourier_fft_length: int = 262_144

    def __post_init__(self) -> None:
        if self.interpolation not in ("linear", "lagrange", "fourier", "thiran"):
            raise ValueError(f"unknown interpolation {self.interpolation!r}")
        if self.realization not in ("df2", "literal", "frequency"):
            raise ValueError(f"unknown realization {self.realization!r}")
        if self.state_policy not in ("hard_reset", "persistent"):
            raise ValueError(f"unknown state policy {self.state_policy!r}")
        if self.sample_rate != SAMPLE_RATE or self.sample_count < 1:
            raise ValueError("the registered renderer is fixed at 4 kHz")
        if (self.loop_gain, self.loop_pole, self.pluck_position) != (0.99, 0.2, 0.23):
            raise ValueError("the paper fixes g=0.99, a1=0.2, and beta=0.23")
        if self.phase_correction_iterations != 12:
            raise ValueError("the registered phase solver uses 12 iterations")
        if self.interpolation == "fourier" and self.realization != "frequency":
            raise ValueError("Fourier propagation requires the FLAMO frequency realization")
        if self.interpolation != "fourier" and self.realization == "frequency":
            raise ValueError("the FLAMO frequency realization requires Fourier interpolation")
        if self.interpolation == "fourier" and self.state_policy != "hard_reset":
            raise ValueError("frequency-sampled propagation has no persistent state")
        if self.interpolation == "fourier" and (
            self.fourier_fft_length % 2
            or self.fourier_fft_length < 2 * self.sample_count
        ):
            raise ValueError(
                "Fourier waveguide propagation requires an even FFT at least "
                "twice the rendered impulse-response horizon"
            )

    @property
    def segment_orders(self) -> tuple[int, int] | None:
        """Long/short rail orders registered for each interpolation family."""
        if self.interpolation == "linear":
            return (1, 1)
        if self.interpolation == "lagrange":
            return (5, 1)
        if self.interpolation == "thiran":
            return (3, 1)
        return None


@dataclass(frozen=True)
class OptimizerConfig:
    """Adam and rollback schedule used by every fit in the paper."""

    learning_rate: float = 0.05
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8
    plateau_patience: int = 100
    stop_patience: int = 250
    lr_factor: float = 0.3
    minimum_learning_rate: float = 1e-5
    meaningful_relative_improvement: float = 1e-4
    maximum_updates: int = 3_000

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0 or self.minimum_learning_rate <= 0.0:
            raise ValueError("learning rates must be positive")
        if not (0.0 < self.lr_factor < 1.0):
            raise ValueError("lr_factor must lie strictly between zero and one")
        if self.plateau_patience < 1 or self.stop_patience <= self.plateau_patience:
            raise ValueError("stop patience must exceed plateau patience")
        if self.maximum_updates < 1:
            raise ValueError("maximum_updates must be positive")


PAPER_OPTIMIZER = OptimizerConfig()


def initial_candidate(cardinality: int, *, device: torch.device | str) -> EventPhrase:
    """Return the one registered equal-cell initialisation."""
    if cardinality not in CARDINALITIES:
        raise ValueError("cardinality must be one of 1, 2, 4, 6, or 8")
    f0 = torch.full((cardinality,), 160.0, dtype=torch.float64, device=device)
    index = torch.arange(cardinality, dtype=torch.float64, device=device)
    onset = 0.2 + (index + 0.5) * (1.6 / cardinality)
    return EventPhrase(f0, onset)


__all__ = [
    "CARDINALITIES",
    "EventPhrase",
    "ExciterConfig",
    "F0_BOUNDS_HZ",
    "InterpolationMethod",
    "MASTER_SEED",
    "ONSET_BOUNDS_SECONDS",
    "OnsetMethod",
    "OptimizerConfig",
    "PAPER_OPTIMIZER",
    "Realization",
    "SAMPLE_COUNT",
    "SAMPLE_RATE",
    "StatePolicy",
    "WaveguideConfig",
    "initial_candidate",
]
