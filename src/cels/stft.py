"""Waveform conveniences for Cumulative Energy Losses."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from .loss import (
    DIAGONAL_DIRECTIONS,
    BoundCumulativeEnergyLoss,
    CumulativeEnergyLoss,
    Direction,
    Reduction,
)


class STFTPower(nn.Module):
    """Periodic-Hann STFT power with frequency and time coordinates."""

    def __init__(
        self,
        *,
        sample_rate: int,
        n_fft: int = 1024,
        hop_length: int = 256,
        center: bool = False,
        normalized: bool = False,
    ) -> None:
        super().__init__()
        if sample_rate <= 0 or n_fft <= 1 or hop_length <= 0:
            raise ValueError("sample_rate, n_fft, and hop_length must be positive")
        if hop_length > n_fft:
            raise ValueError("hop_length must not exceed n_fft")
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.center = bool(center)
        self.normalized = bool(normalized)
        self.register_buffer(
            "window", torch.hann_window(n_fft, periodic=True, dtype=torch.float64)
        )

    def forward(self, audio: Tensor) -> Tensor:
        if not isinstance(audio, Tensor) or audio.ndim < 1 or not audio.is_floating_point():
            raise ValueError("audio must be a floating-point tensor with a sample axis")
        leading = audio.shape[:-1]
        rows = audio.reshape(-1, audio.shape[-1])
        spectrum = torch.stft(
            rows,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(dtype=audio.dtype, device=audio.device),
            center=self.center,
            pad_mode="constant",
            normalized=self.normalized,
            onesided=True,
            return_complex=True,
        )
        return spectrum.abs().square().reshape(
            *leading, spectrum.shape[-2], spectrum.shape[-1]
        )

    def coordinates(self, power: Tensor) -> tuple[Tensor, Tensor]:
        frequency = torch.fft.rfftfreq(
            self.n_fft, d=1.0 / self.sample_rate, device=power.device
        ).to(power.dtype)
        time = torch.arange(power.shape[-1], dtype=power.dtype, device=power.device)
        time = time * (self.hop_length / self.sample_rate)
        return frequency, time


class STFTCumulativeEnergyLoss(nn.Module):
    """Apply a Cumulative Energy Loss to STFT power."""

    def __init__(
        self,
        *,
        sample_rate: int,
        n_fft: int = 1024,
        hop_length: int = 256,
        center: bool = False,
        normalized: bool = False,
        directions: Iterable[Direction | str] = DIAGONAL_DIRECTIONS,
        log_weighting: bool = False,
        time_decay_seconds: float | None = None,
        frequency_decay_hz: float | None = None,
        frequency_floor_hz: float = 20.0,
        eps: float = 1e-12,
        reduction: Reduction = "mean",
    ) -> None:
        super().__init__()
        self.transform = STFTPower(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            center=center,
            normalized=normalized,
        )
        self.map_loss = CumulativeEnergyLoss(
            directions=directions,
            log_weighting=log_weighting,
            time_decay_seconds=time_decay_seconds,
            frequency_decay_hz=frequency_decay_hz,
            frequency_floor_hz=frequency_floor_hz,
            eps=eps,
            reduction=reduction,
        )

    def _represent(self, audio: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        power = self.transform(audio)
        frequency, time = self.transform.coordinates(power)
        return power, frequency, time

    def directional_distances(self, candidate: Tensor, target: Tensor) -> Tensor:
        candidate_power, frequency, time = self._represent(candidate)
        target_power, target_frequency, target_time = self._represent(target)
        if not torch.equal(frequency, target_frequency) or not torch.equal(time, target_time):
            raise ValueError("candidate and target STFT grids differ")
        return self.map_loss.directional_distances(
            candidate_power, target_power, frequency=frequency, time=time
        )

    def forward(self, candidate: Tensor, target: Tensor) -> Tensor:
        candidate_power, frequency, time = self._represent(candidate)
        target_power, target_frequency, target_time = self._represent(target)
        if not torch.equal(frequency, target_frequency) or not torch.equal(time, target_time):
            raise ValueError("candidate and target STFT grids differ")
        return self.map_loss(
            candidate_power, target_power, frequency=frequency, time=time
        )

    def bind(self, target: Tensor) -> BoundSTFTCumulativeEnergyLoss:
        target_power, frequency, time = self._represent(target)
        return BoundSTFTCumulativeEnergyLoss(
            self.transform, self.map_loss.bind(target_power, frequency=frequency, time=time)
        )


class BoundSTFTCumulativeEnergyLoss(nn.Module):
    """An STFT CeL with cached target power and cumulative surfaces."""

    def __init__(self, transform: STFTPower, bound: BoundCumulativeEnergyLoss) -> None:
        super().__init__()
        self.transform = transform
        self.bound = bound

    def directional_distances(self, candidate: Tensor) -> Tensor:
        return self.bound.directional_distances(self.transform(candidate))

    def forward(self, candidate: Tensor) -> Tensor:
        return self.bound(self.transform(candidate))


def paper_loss(variant: str = "cel", *, reduction: Reduction = "mean") -> STFTCumulativeEnergyLoss:
    """Return one of the frozen 16-kHz ICASSP 2027 CeL configurations."""
    key = variant.lower().replace("-", "_")
    common = dict(sample_rate=16_000, n_fft=1024, hop_length=256, reduction=reduction)
    if key == "cel":
        return STFTCumulativeEnergyLoss(**common)
    if key == "log_cel":
        return STFTCumulativeEnergyLoss(**common, log_weighting=True)
    if key == "dec_cel":
        return STFTCumulativeEnergyLoss(
            **common, time_decay_seconds=1.0, frequency_decay_hz=1000.0
        )
    if key == "tlog_cel":
        return STFTCumulativeEnergyLoss(
            **common,
            directions=(Direction.RIGHT_UP, Direction.RIGHT_DOWN),
            log_weighting=True,
        )
    raise ValueError("variant must be cel, log_cel, dec_cel, or tlog_cel")


__all__ = [
    "BoundSTFTCumulativeEnergyLoss",
    "STFTCumulativeEnergyLoss",
    "STFTPower",
    "paper_loss",
]
