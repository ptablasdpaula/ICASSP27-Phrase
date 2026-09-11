"""The six frozen 4-kHz objectives and their versioned registry."""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

import scipy.signal.windows
import torch
from torch import Tensor

LOSS_NAMES: Final = (
    "waveform_l1",
    "waveform_mse",
    "smooth_mss",
    "sot_published_composite",
    "log_jtfot",
    "bidirectional_cumulative_energy",
)
LOSS_LABELS: Final = (
    "L_1", "L_2", "MSS", "SOT", "TFW_2", "BiCuL",
)
PAPER_LOSSES: Final = dict(zip(LOSS_LABELS, LOSS_NAMES, strict=True))
SMOOTH_WINDOWS: Final = (17, 31, 67, 127, 257, 509)
SMOOTH_HOPS: Final = (8, 15, 33, 63, 128, 254)
SOT_MSS_WINDOWS: Final = (512, 256, 128, 64, 32, 16)
SOT_MSS_HOPS: Final = tuple(value // 4 for value in SOT_MSS_WINDOWS)


def _batch(audio: Tensor) -> tuple[Tensor, bool]:
    if audio.dtype != torch.float64:
        raise TypeError("losses require float64 audio")
    if audio.ndim == 1:
        return audio[None, :], True
    if audio.ndim == 2:
        return audio, False
    raise ValueError("audio must have shape [samples] or [batch,samples]")


def _restore(value: Tensor, squeezed: bool) -> Tensor:
    return value[0] if squeezed else value


def periodic_flattop(length: int, reference: Tensor) -> Tensor:
    return torch.as_tensor(
        scipy.signal.windows.flattop(length, sym=False),
        dtype=reference.dtype, device=reference.device)


def _stft(audio: Tensor, *, n_fft: int, hop: int, window: Tensor,
          center: bool = True, pad_mode: str = "constant") -> Tensor:
    return torch.stft(
        audio, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window,
        center=center, pad_mode=pad_mode, normalized=False, onesided=True,
        return_complex=True)


class BoundLoss:
    """Common target-bound interface returning one value per candidate."""

    def distances(self, candidate: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, candidate: Tensor) -> Tensor:
        return self.distances(candidate)


class WaveformDistance(BoundLoss):
    def __init__(self, target: Tensor, *, p: int):
        self.target = target.detach()
        self.p = p

    def distances(self, candidate: Tensor) -> Tensor:
        rows, squeezed = _batch(candidate)
        difference = rows - self.target[None, :]
        value = difference.abs().mean(dim=-1) if self.p == 1 else (
            difference.square().mean(dim=-1))
        return _restore(value, squeezed)


class SmoothMSSDistance(BoundLoss):
    """Time-scaled W_F,S_5,C_2,D_2 Smooth MSS."""

    def __init__(self, target: Tensor, *, sample_rate: int = 4_000):
        self.target = target.detach()
        self.sample_rate = sample_rate
        self.windows: list[Tensor] = []
        self.references: list[Tensor] = []
        for n_fft, hop in zip(SMOOTH_WINDOWS, SMOOTH_HOPS, strict=True):
            window = periodic_flattop(n_fft, target)
            magnitude = _stft(
                target[None], n_fft=n_fft, hop=hop, window=window,
                center=True, pad_mode="reflect").abs()
            self.windows.append(window)
            self.references.append(torch.log1p(magnitude).detach())

    def distances(self, candidate: Tensor) -> Tensor:
        rows, squeezed = _batch(candidate)
        value = rows.new_zeros(rows.shape[0])
        for n_fft, hop, window, target in zip(
                SMOOTH_WINDOWS,
                SMOOTH_HOPS,
                self.windows,
                self.references,
                strict=True,
        ):
            magnitude = _stft(
                rows, n_fft=n_fft, hop=hop, window=window,
                center=True, pad_mode="reflect").abs()
            value = value + (torch.log1p(magnitude) - target).square().sum(
                dim=(-2, -1))
        return _restore(value, squeezed)


def _wasserstein_frequency_rows(x: Tensor, y: Tensor,
                                positions: Tensor) -> Tensor:
    cumulative_x = torch.cumsum(x, dim=-1)
    cumulative_y = torch.cumsum(y, dim=-1)
    quantiles = torch.sort(torch.cat((cumulative_x, cumulative_y), dim=-1),
                           dim=-1).values
    limit = positions.numel() - 1
    index_x = torch.searchsorted(
        cumulative_x.contiguous(), quantiles.contiguous()).clamp(max=limit)
    index_y = torch.searchsorted(
        cumulative_y.contiguous(), quantiles.contiguous()).clamp(max=limit)
    widths = torch.diff(
        quantiles, dim=-1, prepend=torch.zeros_like(quantiles[..., :1]))
    # Published unbalanced SOT cutoff: unmatched target quantiles above the
    # candidate's unit mass make no contribution.
    widths = torch.where(quantiles > 1.0, torch.zeros_like(widths), widths)
    displacement = positions[index_x] - positions[index_y]
    return (widths * displacement.square()).sum(dim=-1)


class PublishedSOTCompositeDistance(BoundLoss):
    """Published asymmetric SOT plus 0.05 linear-magnitude MSS."""

    def __init__(self, target: Tensor, *, sample_rate: int = 4_000):
        self.target = target.detach()
        self.sample_rate = sample_rate
        self.sot_window = periodic_flattop(512, target)
        self.target_power = _stft(
            target[None], n_fft=512, hop=64, window=self.sot_window,
            center=True, pad_mode="reflect").abs().square().detach()
        self.positions = torch.linspace(
            0.0, 1.0, 257, dtype=target.dtype, device=target.device)
        self.mss_windows = [torch.hann_window(
            n_fft, periodic=True, dtype=target.dtype, device=target.device)
            for n_fft in SOT_MSS_WINDOWS]
        self.mss_targets = [
            _stft(target[None], n_fft=n_fft, hop=hop, window=window,
                  center=True, pad_mode="reflect").abs().detach()
            for n_fft, hop, window in zip(
                SOT_MSS_WINDOWS, SOT_MSS_HOPS, self.mss_windows, strict=True)
        ]

    def distances(self, candidate: Tensor) -> Tensor:
        rows, squeezed = _batch(candidate)
        power = _stft(
            rows, n_fft=512, hop=64, window=self.sot_window,
            center=True, pad_mode="reflect").abs().square()
        x = power.transpose(-2, -1)
        y = self.target_power.expand(rows.shape[0], -1, -1).transpose(-2, -1)
        candidate_mass = x.sum(dim=-1, keepdim=True) + 1e-8
        x_weights = x / candidate_mass
        # Deliberately asymmetric, matching the published implementation.
        y_weights = y / candidate_mass
        flat_count = x.shape[0] * x.shape[1]
        sot = _wasserstein_frequency_rows(
            x_weights.reshape(flat_count, -1),
            y_weights.reshape(flat_count, -1), self.positions).reshape(
                x.shape[:2]).mean(dim=-1)
        mss = rows.new_zeros(rows.shape[0])
        for n_fft, hop, window, target in zip(
                SOT_MSS_WINDOWS, SOT_MSS_HOPS,
                self.mss_windows, self.mss_targets, strict=True):
            magnitude = _stft(
                rows, n_fft=n_fft, hop=hop, window=window,
                center=True, pad_mode="reflect").abs()
            # DDSP's linear-magnitude MSS term is an elementwise mean L1.
            mss = mss + (magnitude - target).abs().mean(dim=(-2, -1))
        return _restore(sot + 0.05 * mss, squeezed)


@lru_cache(maxsize=32)
def _projection_geometry(frequency_bins: int, time_bins: int,
                         sample_rate: int, n_fft: int, hop: int,
                         device: str, dtype_name: str) -> tuple[Tensor, Tensor]:
    dtype = getattr(torch, dtype_name)
    frequency = torch.arange(frequency_bins, dtype=dtype, device=device) \
        * (float(sample_rate) / float(n_fft))
    frequency = torch.log2(torch.clamp(frequency, min=20.0) / 20.0)
    time = torch.arange(time_bins, dtype=dtype, device=device) \
        * (float(hop) / float(sample_rate))
    fcoord = frequency[:, None].expand(-1, time_bins).reshape(-1)
    tcoord = time[None, :].expand(frequency_bins, -1).reshape(-1)
    orders, positions = [], []
    for index in range(10):
        angle = 2.0 * math.pi * index / 10.0
        coordinate = tcoord * math.cos(angle) + fcoord * math.sin(angle)
        order = torch.argsort(coordinate, stable=True)
        orders.append(order)
        positions.append(coordinate[order])
    return torch.stack(orders), torch.stack(positions)


def _wasserstein_projected(x: Tensor, y: Tensor, positions: Tensor) -> Tensor:
    cumulative_x = torch.cumsum(x, dim=-1)
    cumulative_y = torch.cumsum(y, dim=-1)
    cumulative_x = cumulative_x / cumulative_x[..., -1:]
    cumulative_y = cumulative_y / cumulative_y[..., -1:]
    quantiles = torch.sort(torch.cat((cumulative_x, cumulative_y), dim=-1),
                           dim=-1).values
    limit = positions.shape[-1] - 1
    index_x = torch.searchsorted(
        cumulative_x.contiguous(), quantiles.contiguous()).clamp(max=limit)
    index_y = torch.searchsorted(
        cumulative_y.contiguous(), quantiles.contiguous()).clamp(max=limit)
    widths = torch.diff(
        quantiles, dim=-1, prepend=torch.zeros_like(quantiles[..., :1]))
    support = positions
    while support.ndim < index_x.ndim:
        support = support.unsqueeze(0)
    support = support.expand(*index_x.shape[:-1], support.shape[-1])
    displacement = (torch.gather(support, -1, index_x)
                    - torch.gather(support, -1, index_y))
    return torch.sqrt((widths * displacement.square()).sum(dim=-1).clamp_min(0.0))


class LogJTFOTDistance(BoundLoss):
    """This paper's log-frequency adaptation of Fabiani et al.'s JTFOT."""

    def __init__(self, target: Tensor, *, sample_rate: int = 4_000):
        self.target = target.detach()
        self.sample_rate = sample_rate
        self.n_fft = 256
        self.hop = 128
        self.window = torch.hann_window(
            self.n_fft, periodic=True, dtype=target.dtype,
            device=target.device)
        magnitude = _stft(
            target[None], n_fft=self.n_fft, hop=self.hop,
            window=self.window, center=True, pad_mode="constant").abs()
        self.target_magnitude = magnitude.detach()

    def distances(self, candidate: Tensor) -> Tensor:
        rows, squeezed = _batch(candidate)
        magnitude = _stft(
            rows, n_fft=self.n_fft, hop=self.hop, window=self.window,
            center=True, pad_mode="constant").abs()
        target = self.target_magnitude.expand(rows.shape[0], -1, -1)
        mass_x = magnitude.sum(dim=(-2, -1))
        mass_y = target.sum(dim=(-2, -1))
        if bool((mass_x.detach() <= 1e-12).any()) or bool(
                (mass_y.detach() <= 1e-12).any()):
            raise ValueError("Log-JTFOT is undefined on silence")
        order, positions = _projection_geometry(
            magnitude.shape[-2], magnitude.shape[-1], self.sample_rate,
            self.n_fft, self.hop, str(rows.device),
            str(rows.dtype).removeprefix("torch."))
        x = magnitude.reshape(rows.shape[0], -1) / mass_x[:, None]
        y = target.reshape(rows.shape[0], -1) / mass_y[:, None]
        value = _wasserstein_projected(x[:, order], y[:, order], positions).mean(dim=-1)
        return _restore(value, squeezed)


def reverse_cumsum(value: Tensor, dim: int) -> Tensor:
    return torch.flip(torch.cumsum(torch.flip(value, (dim,)), dim=dim), (dim,))


class BidirectionalCumulativeEnergyDistance(BoundLoss):
    """Four directional cumulative-power surfaces with target normalization."""

    def __init__(self, target: Tensor, *, sample_rate: int = 4_000):
        self.target = target.detach()
        self.sample_rate = sample_rate
        self.n_fft = 256
        self.hop = 64
        self.sqrt_floor = 1e-12
        self.window = torch.hann_window(
            self.n_fft, periodic=True, dtype=target.dtype,
            device=target.device)
        self.references: list[tuple[Tensor, Tensor]] = []
        with torch.no_grad():
            power = self._power(target[None])[0]
            for time_reverse in (False, True):
                time_surface = (reverse_cumsum(power, 1) if time_reverse
                                else torch.cumsum(power, dim=1))
                for frequency_reverse in (False, True):
                    surface = (reverse_cumsum(time_surface, 0)
                               if frequency_reverse
                               else torch.cumsum(time_surface, dim=0))
                    scale = surface.amax().clamp_min(torch.finfo(surface.dtype).tiny)
                    self.references.append((
                        torch.sqrt((surface / scale).clamp_min(
                            self.sqrt_floor)).detach(),
                        scale.detach()))

    def _power(self, rows: Tensor) -> Tensor:
        power = _stft(
            rows, n_fft=self.n_fft, hop=self.hop, window=self.window,
            center=False, pad_mode="constant").abs().square()
        return power

    def distances(self, candidate: Tensor) -> Tensor:
        rows, squeezed = _batch(candidate)
        power = self._power(rows)
        terms = []
        reference_index = 0
        for time_reverse in (False, True):
            time_surface = (reverse_cumsum(power, 2) if time_reverse
                            else torch.cumsum(power, dim=2))
            for frequency_reverse in (False, True):
                surface = (reverse_cumsum(time_surface, 1)
                           if frequency_reverse
                           else torch.cumsum(time_surface, dim=1))
                target, scale = self.references[reference_index]
                reference_index += 1
                difference = torch.sqrt(
                    (surface / scale).clamp_min(self.sqrt_floor)) - target[None]
                terms.append(torch.linalg.vector_norm(
                    difference.flatten(start_dim=1), dim=1)
                    / math.sqrt(difference.shape[-2] * difference.shape[-1]))
        value = torch.stack(terms, dim=-1).mean(dim=-1)
        return _restore(value, squeezed)


@dataclass(frozen=True)
class LossRegistry:
    schema: str = "loss-registry-v1"
    names: tuple[str, ...] = LOSS_NAMES

    def build(self, name: str, target: Tensor) -> BoundLoss:
        name = canonical_loss_name(name)
        if name not in self.names:
            raise ValueError(f"unknown loss {name!r}")
        if name == "waveform_l1":
            return WaveformDistance(target, p=1)
        if name == "waveform_mse":
            return WaveformDistance(target, p=2)
        if name == "smooth_mss":
            return SmoothMSSDistance(target)
        if name == "sot_published_composite":
            return PublishedSOTCompositeDistance(target)
        if name == "log_jtfot":
            return LogJTFOTDistance(target)
        return BidirectionalCumulativeEnergyDistance(target)


REGISTRY = LossRegistry()


def canonical_loss_name(name: str) -> str:
    """Map paper labels and friendly spellings to the frozen implementation."""
    normalized = name.strip().lower().replace("mathcal", "").replace("_", "")
    aliases = {
        "l1": "waveform_l1",
        "waveforml1": "waveform_l1",
        "l2": "waveform_mse",
        "waveforml2": "waveform_mse",
        "waveformmse": "waveform_mse",
        "mss": "smooth_mss",
        "smoothmss": "smooth_mss",
        "sot": "sot_published_composite",
        "sotpublishedcomposite": "sot_published_composite",
        "tfw2": "log_jtfot",
        "logjtfot": "log_jtfot",
        "bicul": "bidirectional_cumulative_energy",
        "bidirectionalcumulativeenergy": "bidirectional_cumulative_energy",
    }
    if name in LOSS_NAMES:
        return name
    try:
        return aliases[normalized]
    except KeyError as error:
        choices = ", ".join(LOSS_LABELS)
        raise ValueError(f"unknown loss {name!r}; choose from {choices}") from error


def build_loss(name: str, target: Tensor) -> BoundLoss:
    return REGISTRY.build(name, target)


__all__ = [
    "BidirectionalCumulativeEnergyDistance",
    "LOSS_LABELS",
    "LOSS_NAMES",
    "PAPER_LOSSES",
    "LogJTFOTDistance",
    "LossRegistry",
    "PublishedSOTCompositeDistance",
    "REGISTRY",
    "SMOOTH_HOPS",
    "SMOOTH_WINDOWS",
    "SOT_MSS_HOPS",
    "SOT_MSS_WINDOWS",
    "SmoothMSSDistance",
    "WaveformDistance",
    "build_loss",
    "canonical_loss_name",
    "periodic_flattop",
    "reverse_cumsum",
]
