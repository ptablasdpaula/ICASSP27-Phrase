"""The eight paper 4-kHz objectives and their versioned registry."""
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
    "linear_jtfot",
    "log_jtfot",
    "bidirectional_cumulative_energy",
    "log_quadrature_bicul",
)
LOSS_LABELS: Final = (
    "L_1", "L_2", "MSS", "SOT", "TFW_2", "TFW_2 (1s=1oct)",
    "BiCuL", "LogQ_BiCuL",
)
PAPER_LOSSES: Final = dict(zip(LOSS_LABELS, LOSS_NAMES, strict=True))
SMOOTH_WINDOWS: Final = (17, 31, 67, 127, 257, 509)
SMOOTH_HOPS: Final = (8, 15, 33, 63, 128, 254)
SOT_MSS_WINDOWS: Final = (512, 256, 128, 64, 32, 16)
SOT_MSS_HOPS: Final = tuple(value // 4 for value in SOT_MSS_WINDOWS)
FABIANI_TIME_SCALE_HZ_PER_SECOND: Final = 1_000.0


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


@lru_cache(maxsize=32)
def _linear_projection_geometry(
        frequency_bins: int, time_bins: int, sample_rate: int, n_fft: int,
        hop: int, device: str, dtype_name: str) -> tuple[Tensor, Tensor]:
    """Fabiani et al.'s linear-Hz geometry with sqrt(k)=1000 Hz/s."""
    if (frequency_bins < 2 or time_bins < 2 or sample_rate != 4_000
            or n_fft != 256 or hop != 128):
        raise ValueError("linear JTFOT received an unregistered STFT geometry")
    dtype = getattr(torch, dtype_name)
    frequency = torch.arange(
        frequency_bins, dtype=dtype, device=device
    ) * (float(sample_rate) / float(n_fft))
    time = torch.arange(
        time_bins, dtype=dtype, device=device
    ) * (float(hop) / float(sample_rate)) * FABIANI_TIME_SCALE_HZ_PER_SECOND
    fcoord = frequency[:, None].expand(-1, time_bins).reshape(-1)
    tcoord = time[None, :].expand(frequency_bins, -1).reshape(-1)
    orders, positions = [], []
    for index in range(10):
        angle = 2.0 * math.pi * index / 10.0
        coordinate = tcoord * math.cos(angle) + fcoord * math.sin(angle)
        order = torch.argsort(coordinate, stable=True)
        orders.append(order)
        positions.append(coordinate[order])
    result = torch.stack(orders), torch.stack(positions)
    if not all(bool(torch.isfinite(value).all()) for value in result):
        raise FloatingPointError("linear JTFOT geometry is non-finite")
    return result


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


class LinearJTFOTDistance(BoundLoss):
    """Published JTFOT: physical hertz with one second equal to 1000 Hz."""

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
            raise ValueError("linear JTFOT is undefined on silence")
        order, positions = _linear_projection_geometry(
            magnitude.shape[-2], magnitude.shape[-1], self.sample_rate,
            self.n_fft, self.hop, str(rows.device),
            str(rows.dtype).removeprefix("torch."))
        x = magnitude.reshape(rows.shape[0], -1) / mass_x[:, None]
        y = target.reshape(rows.shape[0], -1) / mass_y[:, None]
        value = _wasserstein_projected(
            x[:, order], y[:, order], positions).mean(dim=-1)
        if not bool(torch.isfinite(value.detach()).all()):
            raise FloatingPointError("linear JTFOT produced a non-finite loss")
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


CEL_DIRECTIONS: Final = ("right_up", "right_down", "left_up", "left_down")
CEL_NAMES: Final = tuple(
    f"cel_{mask:02d}{'_lw' if weighted else ''}"
    for weighted in (False, True) for mask in range(1, 16)
)


class CumulativeEnergyDistance(BidirectionalCumulativeEnergyDistance):
    """Any nonempty direction subset, with optional Log-Weighing.

    Direction order is (right/up, right/down, left/up, left/down).
    The feature remains sqrt(max(normalised power, 1e-12)). Legacy classes
    remain available for reproducing archived experiments.
    """

    def __init__(self, target: Tensor, *, directions: tuple[str, ...] = CEL_DIRECTIONS,
                 log_weighing: bool = False, sample_rate: int = 4_000):
        if (not directions or len(set(directions)) != len(directions)
                or any(d not in CEL_DIRECTIONS for d in directions)):
            raise ValueError("select distinct, nonempty CeL directions")
        super().__init__(target, sample_rate=sample_rate)
        self.directions = tuple(d for d in CEL_DIRECTIONS if d in directions)
        self.indices = tuple(CEL_DIRECTIONS.index(d) for d in self.directions)
        self.log_weighing = log_weighing
        shape = self.references[0][0].shape
        f = torch.arange(shape[0], dtype=target.dtype, device=target.device)
        f = torch.log2((f * sample_rate / self.n_fft).clamp_min(20) / 20)
        t = torch.arange(shape[1], dtype=target.dtype, device=target.device)
        t = t * self.hop / sample_rate
        weights = []
        for tr in (False, True):
            for fr in (False, True):
                widths = []
                for axis, reverse in ((f, fr), (t, tr)):
                    gap = torch.diff(axis)
                    zero = axis.new_zeros(1)
                    widths.append(torch.cat((zero, gap) if reverse else (gap, zero)))
                weight = widths[0][:, None] * widths[1][None, :]
                if not bool(weight.sum() > 0):
                    raise ValueError("Log-Weighing requires positive grid area")
                weights.append(torch.sqrt(weight / weight.sum()))
        self.sqrt_weights = torch.stack(weights)

    def directional_distances(self, candidate: Tensor) -> Tensor:
        """Return [...,2,4] elementary RMSs: uniform, then Log-Weighed."""
        rows, squeezed = _batch(candidate)
        power = self._power(rows)
        ordinary, weighted = [], []
        for index, (tr, fr) in enumerate(
                 (tr, fr) for tr in (False, True) for fr in (False, True)):
            surface = reverse_cumsum(power, 2) if tr else power.cumsum(2)
            surface = reverse_cumsum(surface, 1) if fr else surface.cumsum(1)
            reference, mass = self.references[index]
            error = torch.sqrt((surface / mass).clamp_min(self.sqrt_floor)) - reference
            ordinary.append(torch.linalg.vector_norm(error.flatten(1), dim=1)
                            / math.sqrt(error.shape[1] * error.shape[2]))
            weighted.append(torch.linalg.vector_norm(
                (error * self.sqrt_weights[index]).flatten(1), dim=1))
        values = torch.stack((torch.stack(ordinary, -1), torch.stack(weighted, -1)), -2)
        return _restore(values, squeezed)

    def distances(self, candidate: Tensor) -> Tensor:
        return self.directional_distances(candidate)[..., int(self.log_weighing),
                                                       self.indices].mean(-1)


class LogQuadratureBiCumulativeEnergyDistance(BoundLoss):
    """BiCuL with its final RMS integrated over seconds and octaves.

    The four cumulative-power surfaces, target-maximum normalization, and
    square-root feature are identical to canonical BiCuL. Only the final grid
    norm changes: exact adjacent-coordinate widths in seconds and
    log2-frequency provide the quadrature weights. The loss uses no event
    parameters, pitch/onset detectors, or salience weighting.
    """

    def __init__(self, target: Tensor, *, sample_rate: int = 4_000):
        rows, squeezed = _batch(target)
        if not squeezed:
            raise ValueError("log-quadrature BiCuL requires one target vector")
        self.target = target.detach()
        self.sample_rate = sample_rate
        self.n_fft = 256
        self.hop = 64
        self.sqrt_floor = 1e-12
        self.window = torch.hann_window(
            self.n_fft, periodic=True, dtype=target.dtype,
            device=target.device)

        with torch.no_grad():
            power = self._power(rows)[0]
            frequency = torch.arange(
                power.shape[0], dtype=target.dtype, device=target.device
            ) * (float(sample_rate) / float(self.n_fft))
            frequency = torch.log2(torch.clamp(frequency, min=20.0) / 20.0)
            time = torch.arange(
                power.shape[1], dtype=target.dtype, device=target.device
            ) * (float(self.hop) / float(sample_rate))
            frequency_gap = torch.diff(frequency)
            time_gap = torch.diff(time)
            self.frequency_widths = (
                torch.cat((frequency_gap, torch.zeros_like(frequency_gap[:1]))),
                torch.cat((torch.zeros_like(frequency_gap[:1]), frequency_gap)),
            )
            self.time_widths = (
                torch.cat((time_gap, torch.zeros_like(time_gap[:1]))),
                torch.cat((torch.zeros_like(time_gap[:1]), time_gap)),
            )
            widths = (*self.frequency_widths, *self.time_widths)
            if any(
                not bool(torch.isfinite(value).all())
                or bool((value < 0.0).any())
                for value in widths
            ):
                raise FloatingPointError("invalid log-quadrature coordinate widths")

            self.references: list[tuple[Tensor, Tensor]] = []
            for time_reverse in (False, True):
                time_surface = (
                    reverse_cumsum(power, 1)
                    if time_reverse else torch.cumsum(power, dim=1)
                )
                for frequency_reverse in (False, True):
                    surface = (
                        reverse_cumsum(time_surface, 0)
                        if frequency_reverse else torch.cumsum(time_surface, dim=0)
                    )
                    scale = surface.amax().clamp_min(torch.finfo(surface.dtype).tiny)
                    reference = torch.sqrt(
                        (surface / scale).clamp_min(self.sqrt_floor)
                    )
                    self.references.append((reference.detach(), scale.detach()))

    def _power(self, rows: Tensor) -> Tensor:
        return _stft(
            rows, n_fft=self.n_fft, hop=self.hop, window=self.window,
            center=False, pad_mode="constant").abs().square()

    def distances(self, candidate: Tensor) -> Tensor:
        rows, squeezed = _batch(candidate)
        power = self._power(rows)
        terms: list[Tensor] = []
        reference_index = 0
        for time_index, time_reverse in enumerate((False, True)):
            time_surface = (
                reverse_cumsum(power, 2)
                if time_reverse else torch.cumsum(power, dim=2)
            )
            time_width = self.time_widths[time_index]
            for frequency_index, frequency_reverse in enumerate((False, True)):
                surface = (
                    reverse_cumsum(time_surface, 1)
                    if frequency_reverse else torch.cumsum(time_surface, dim=1)
                )
                reference, scale = self.references[reference_index]
                reference_index += 1
                frequency_width = self.frequency_widths[frequency_index]
                weight = frequency_width[:, None] * time_width[None, :]
                weight_sum = weight.sum()
                if not bool(weight_sum > 0.0):
                    raise ValueError("log-quadrature surface has zero area")
                difference = torch.sqrt(
                    (surface / scale).clamp_min(self.sqrt_floor)
                ) - reference[None]
                terms.append(torch.sqrt(
                    (difference.square() * weight[None]).sum(dim=(1, 2))
                    / weight_sum
                ))
        value = torch.stack(terms, dim=-1).mean(dim=-1)
        if not bool(torch.isfinite(value.detach()).all()):
            raise FloatingPointError("log-quadrature BiCuL is non-finite")
        return _restore(value, squeezed)


@dataclass(frozen=True)
class LossRegistry:
    schema: str = "loss-registry-v3"
    names: tuple[str, ...] = LOSS_NAMES + CEL_NAMES

    def build(self, name: str, target: Tensor) -> BoundLoss:
        name = canonical_loss_name(name)
        if name not in self.names:
            raise ValueError(f"unknown loss {name!r}")
        if name in CEL_NAMES:
            mask = int(name.split("_")[1])
            return CumulativeEnergyDistance(
                target, directions=tuple(d for i, d in enumerate(CEL_DIRECTIONS)
                                         if mask & (1 << i)),
                log_weighing=name.endswith("_lw"))
        if name == "waveform_l1":
            return WaveformDistance(target, p=1)
        if name == "waveform_mse":
            return WaveformDistance(target, p=2)
        if name == "smooth_mss":
            return SmoothMSSDistance(target)
        if name == "sot_published_composite":
            return PublishedSOTCompositeDistance(target)
        if name == "linear_jtfot":
            return LinearJTFOTDistance(target)
        if name == "log_jtfot":
            return LogJTFOTDistance(target)
        if name == "bidirectional_cumulative_energy":
            return BidirectionalCumulativeEnergyDistance(target)
        return LogQuadratureBiCumulativeEnergyDistance(target)


REGISTRY = LossRegistry()


def canonical_loss_name(name: str) -> str:
    """Map paper labels and friendly spellings to the frozen implementation."""
    if name in PAPER_LOSSES:
        return PAPER_LOSSES[name]
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
        "tfw2": "linear_jtfot",
        "linearjtfot": "linear_jtfot",
        "tfw2 (1s=1oct)": "log_jtfot",
        "tfw21s1oct": "log_jtfot",
        "logjtfot": "log_jtfot",
        "bicul": "bidirectional_cumulative_energy",
        "bidirectionalcumulativeenergy": "bidirectional_cumulative_energy",
        "logqbicul": "log_quadrature_bicul",
        "logquadraturebicul": "log_quadrature_bicul",
    }
    if name in LOSS_NAMES or name in CEL_NAMES:
        return name
    try:
        return aliases[normalized]
    except KeyError as error:
        choices = ", ".join(LOSS_LABELS)
        raise ValueError(f"unknown loss {name!r}; choose from {choices}") from error


def build_loss(name: str, target: Tensor) -> BoundLoss:
    return REGISTRY.build(name, target)


__all__ = [
    "CEL_DIRECTIONS",
    "CEL_NAMES",
    "CumulativeEnergyDistance",
    "BidirectionalCumulativeEnergyDistance",
    "FABIANI_TIME_SCALE_HZ_PER_SECOND",
    "LOSS_LABELS",
    "LOSS_NAMES",
    "PAPER_LOSSES",
    "LogJTFOTDistance",
    "LinearJTFOTDistance",
    "LogQuadratureBiCumulativeEnergyDistance",
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
