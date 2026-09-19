"""Frozen 16-kHz objectives used by both paper experiments."""

from __future__ import annotations

import math
from collections.abc import Iterable
from functools import lru_cache
from typing import Final

import scipy.signal.windows
import torch
import torch.nn.functional as F
from cels import CumulativeEnergyLoss, Direction, STFTPower
from cels.loss import BoundCumulativeEnergyLoss
from torch import Tensor

from .config import SAMPLE_RATE

NAMES: Final = (
    "waveform_l1",
    "waveform_mse",
    "single_stft",
    "mss",
    "smooth_mss",
    "sot_published_composite",
    "sot_no_cut",
    "linear_jtfot",
    "log_jtfot",
    "cel",
    "log_cel",
    "dec_cel",
    "tlog_cel",
)
LABELS: Final = (
    "L1",
    "L2",
    "SS",
    "MSS",
    "SmoMSS",
    "SOT",
    "SOT-NC",
    "TFW2",
    "logTFW2",
    "CeL",
    "logCeL",
    "decCeL",
    "tlogCeL",
)

CEL_N_FFT: Final = 1024
CEL_HOP: Final = 256
DDSP_FFT_SIZES: Final = (2048, 1024, 512, 256, 128, 64)
DDSP_HOPS: Final = tuple(size // 4 for size in DDSP_FFT_SIZES)
SMOOTH_WINDOWS: Final = (67, 127, 257, 509, 1021, 2053)
SMOOTH_HOPS: Final = (32, 63, 128, 254, 510, 1026)
SOT_N_FFT: Final = 2048
SOT_HOP: Final = 256
TFW_N_FFT: Final = 1024
TFW_HOP: Final = 512
TFW_PROJECTIONS: Final = 10
TFW_HZ_PER_SECOND: Final = 1000.0


def _rows(audio: Tensor) -> Tensor:
    if not isinstance(audio, Tensor) or not audio.is_floating_point():
        raise TypeError("audio must be a floating-point tensor")
    if audio.ndim == 1:
        return audio[None]
    if audio.ndim != 2:
        raise ValueError("audio must have shape [samples] or [batch,samples]")
    return audio


def _stft(
    audio: Tensor,
    *,
    n_fft: int,
    hop: int,
    window: Tensor,
    center: bool,
    normalized: bool = False,
    pad_mode: str = "constant",
) -> Tensor:
    return torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=center,
        pad_mode=pad_mode,
        normalized=normalized,
        onesided=True,
        return_complex=True,
    )


def _ddsp_magnitude(audio: Tensor, n_fft: int, window: Tensor) -> Tensor:
    """Magenta DDSP STFT: 75% overlap, end padding, no centring."""
    hop = n_fft // 4
    frames = math.ceil(audio.shape[-1] / hop)
    padding = max(0, n_fft + hop * (frames - 1) - audio.shape[-1])
    padded = F.pad(audio, (0, padding))
    return _stft(
        padded,
        n_fft=n_fft,
        hop=hop,
        window=window,
        center=False,
        normalized=True,
    ).abs()


def _safe_log(value: Tensor, eps: float = 1e-5) -> Tensor:
    return torch.log(value.clamp_min(eps))


def _quantile_wasserstein(
    x: Tensor,
    y: Tensor,
    positions: Tensor,
    *,
    limit_quantile_range: bool,
) -> Tensor:
    """Source-equivalent discrete 1-D W2^2 for already scaled weights."""
    cumulative_x = torch.cumsum(x, dim=-1)
    cumulative_y = torch.cumsum(y, dim=-1)
    quantiles = torch.sort(torch.cat((cumulative_x, cumulative_y), dim=-1), dim=-1).values
    last = positions.numel() - 1
    index_x = torch.searchsorted(cumulative_x.contiguous(), quantiles.contiguous()).clamp(max=last)
    index_y = torch.searchsorted(cumulative_y.contiguous(), quantiles.contiguous()).clamp(max=last)
    widths = torch.diff(
        quantiles, dim=-1, prepend=torch.zeros_like(quantiles[..., :1])
    )
    if limit_quantile_range:
        widths = torch.where(quantiles > 1.0, torch.zeros_like(widths), widths)
    displacement = positions[index_x] - positions[index_y]
    return (widths * displacement.square()).sum(dim=-1)


@lru_cache(maxsize=32)
def _tfw_projection_geometry(
    frequency_bins: int,
    time_bins: int,
    logarithmic_frequency: bool,
    device: str,
    dtype_name: str,
) -> tuple[Tensor, Tensor]:
    dtype = getattr(torch, dtype_name)
    frequency = torch.arange(frequency_bins, dtype=dtype, device=device) * (
        SAMPLE_RATE / TFW_N_FFT
    )
    if logarithmic_frequency:
        frequency = torch.log2(frequency.clamp_min(20.0) / 20.0)
        time = torch.arange(time_bins, dtype=dtype, device=device) * (TFW_HOP / SAMPLE_RATE)
    else:
        time = (
            torch.arange(time_bins, dtype=dtype, device=device)
            * (TFW_HOP / SAMPLE_RATE)
            * TFW_HZ_PER_SECOND
        )
    frequency_grid = frequency[:, None].expand(-1, time_bins).reshape(-1)
    time_grid = time[None].expand(frequency_bins, -1).reshape(-1)
    orders, positions = [], []
    for index in range(TFW_PROJECTIONS):
        angle = 2.0 * math.pi * index / TFW_PROJECTIONS
        coordinate = time_grid * math.cos(angle) + frequency_grid * math.sin(angle)
        order = torch.argsort(coordinate, stable=True)
        orders.append(order)
        positions.append(coordinate[order])
    return torch.stack(orders), torch.stack(positions)


def _projected_wasserstein(x: Tensor, y: Tensor, positions: Tensor) -> Tensor:
    cumulative_x = x.cumsum(-1) / x.sum(-1, keepdim=True)
    cumulative_y = y.cumsum(-1) / y.sum(-1, keepdim=True)
    quantiles = torch.sort(torch.cat((cumulative_x, cumulative_y), -1), -1).values
    last = positions.shape[-1] - 1
    index_x = torch.searchsorted(cumulative_x.contiguous(), quantiles.contiguous()).clamp(max=last)
    index_y = torch.searchsorted(cumulative_y.contiguous(), quantiles.contiguous()).clamp(max=last)
    widths = torch.diff(quantiles, dim=-1, prepend=torch.zeros_like(quantiles[..., :1]))
    support = positions
    while support.ndim < index_x.ndim:
        support = support.unsqueeze(0)
    support = support.expand(*index_x.shape[:-1], support.shape[-1])
    displacement = torch.gather(support, -1, index_x) - torch.gather(support, -1, index_y)
    tiny = torch.finfo(widths.dtype).tiny
    sqrt_widths = torch.where(widths > 0.0, torch.sqrt(widths.clamp_min(tiny)), 0.0)
    return torch.linalg.vector_norm(sqrt_widths * displacement, dim=-1)


class PaperObjectives:
    """Target-bound, source-faithful objectives with shared transforms."""

    def __init__(self, target: Tensor, names: Iterable[str] = NAMES) -> None:
        self.target = _rows(target).detach()
        self.names = tuple(names)
        unknown = set(self.names) - set(NAMES)
        if unknown:
            raise ValueError(f"unknown paper objectives: {sorted(unknown)}")
        self._selected_indices: Tensor | None = None
        self._selected_cels: dict[str, BoundCumulativeEnergyLoss] = {}
        reference = self.target
        dtype, device = reference.dtype, reference.device

        self.cel_transform = STFTPower(
            sample_rate=SAMPLE_RATE,
            n_fft=CEL_N_FFT,
            hop_length=CEL_HOP,
            center=False,
        ).to(device=device, dtype=dtype)
        needs_cel = any(
            name in self.names
            for name in ("single_stft", "cel", "log_cel", "dec_cel", "tlog_cel")
        )
        if needs_cel:
            with torch.no_grad():
                self.cel_target_power = self.cel_transform(reference)
                frequency, time = self.cel_transform.coordinates(self.cel_target_power)
            configurations = {
                "cel": {},
                "log_cel": {"log_weighting": True},
                "dec_cel": {"time_decay_seconds": 1.0, "frequency_decay_hz": 1000.0},
                "tlog_cel": {
                    "directions": (Direction.RIGHT_UP, Direction.RIGHT_DOWN),
                    "log_weighting": True,
                },
            }
            self.cel_sources: dict[str, CumulativeEnergyLoss] = {}
            self.cel_bounds: dict[str, BoundCumulativeEnergyLoss] = {}
            for name, kwargs in configurations.items():
                if name in self.names:
                    source = CumulativeEnergyLoss(reduction="none", **kwargs)
                    self.cel_sources[name] = source
                    self.cel_bounds[name] = source.bind(
                        self.cel_target_power, frequency=frequency, time=time
                    )

        if "mss" in self.names or any(name.startswith("sot_") for name in self.names):
            self.ddsp_windows = [
                torch.hann_window(size, periodic=True, dtype=dtype, device=device)
                for size in DDSP_FFT_SIZES
            ]
            with torch.no_grad():
                self.ddsp_targets = [
                    _ddsp_magnitude(reference, size, window)
                    for size, window in zip(DDSP_FFT_SIZES, self.ddsp_windows, strict=True)
                ]

        if "smooth_mss" in self.names:
            self.smooth_windows = [
                torch.as_tensor(
                    scipy.signal.windows.flattop(size, sym=False), dtype=dtype, device=device
                )
                for size in SMOOTH_WINDOWS
            ]
            with torch.no_grad():
                self.smooth_targets = [
                    torch.log1p(
                        _stft(
                            reference,
                            n_fft=size,
                            hop=hop,
                            window=window,
                            center=True,
                        ).abs()
                    )
                    for size, hop, window in zip(
                        SMOOTH_WINDOWS, SMOOTH_HOPS, self.smooth_windows, strict=True
                    )
                ]

        if any(name.startswith("sot_") for name in self.names):
            self.sot_window = torch.as_tensor(
                scipy.signal.windows.flattop(SOT_N_FFT, sym=False), dtype=dtype, device=device
            )
            with torch.no_grad():
                self.sot_target_power = _stft(
                    reference,
                    n_fft=SOT_N_FFT,
                    hop=SOT_HOP,
                    window=self.sot_window,
                    center=True,
                ).abs().square()
            self.sot_positions = torch.linspace(
                0.0,
                1.0,
                self.sot_target_power.shape[-2],
                dtype=dtype,
                device=device,
            )

        if "linear_jtfot" in self.names or "log_jtfot" in self.names:
            self.tfw_window = torch.hann_window(
                TFW_N_FFT, periodic=True, dtype=dtype, device=device
            )
            with torch.no_grad():
                self.tfw_target = _stft(
                    reference,
                    n_fft=TFW_N_FFT,
                    hop=TFW_HOP,
                    window=self.tfw_window,
                    center=True,
                ).abs()
            dtype_name = str(dtype).removeprefix("torch.")
            for name, logarithmic in (("linear_jtfot", False), ("log_jtfot", True)):
                if name in self.names:
                    order, positions = _tfw_projection_geometry(
                        self.tfw_target.shape[-2],
                        self.tfw_target.shape[-1],
                        logarithmic,
                        str(device),
                        dtype_name,
                    )
                    setattr(self, f"{name}_order", order)
                    setattr(self, f"{name}_positions", positions)

    @staticmethod
    def _select(value: Tensor, indices: Tensor | None) -> Tensor:
        return value if indices is None else value[indices]

    def _cel_bound(self, name: str, indices: Tensor | None) -> BoundCumulativeEnergyLoss:
        bound = self.cel_bounds[name]
        if indices is None:
            return bound
        if self._selected_indices is None or not torch.equal(indices, self._selected_indices):
            self._selected_indices = indices.detach().clone()
            self._selected_cels = {}
        if name not in self._selected_cels:
            self._selected_cels[name] = BoundCumulativeEnergyLoss(
                self.cel_sources[name],
                bound.reference[indices],
                bound.mass[indices],
                bound._geometry(),
            )
        return self._selected_cels[name]

    def values(self, candidate: Tensor, indices: Tensor | None = None) -> dict[str, Tensor]:
        rows = _rows(candidate)
        target = self._select(self.target, indices)
        if rows.shape[-1] != target.shape[-1] or target.shape[0] not in (1, rows.shape[0]):
            raise ValueError("candidate rows and selected targets must be batch-compatible")
        values: dict[str, Tensor] = {}
        difference = rows - target
        if "waveform_l1" in self.names:
            values["waveform_l1"] = difference.abs().mean(-1)
        if "waveform_mse" in self.names:
            values["waveform_mse"] = difference.square().mean(-1)

        cel_names = tuple(
            name for name in ("cel", "log_cel", "dec_cel", "tlog_cel") if name in self.names
        )
        if "single_stft" in self.names or cel_names:
            power = self.cel_transform(rows)
            if "single_stft" in self.names:
                reference = self._select(self.cel_target_power, indices)
                values["single_stft"] = (power.sqrt() - reference.sqrt()).abs().mean((-2, -1))
            for name in cel_names:
                values[name] = self._cel_bound(name, indices)(power)

        ddsp_linear = rows.new_zeros(len(rows))
        if "mss" in self.names or any(name.startswith("sot_") for name in self.names):
            ddsp_log = rows.new_zeros(len(rows))
            for size, window, saved in zip(
                DDSP_FFT_SIZES, self.ddsp_windows, self.ddsp_targets, strict=True
            ):
                magnitude = _ddsp_magnitude(rows, size, window)
                reference = self._select(saved, indices)
                ddsp_linear = ddsp_linear + (magnitude - reference).abs().mean((-2, -1))
                if "mss" in self.names:
                    ddsp_log = ddsp_log + (
                        _safe_log(magnitude) - _safe_log(reference)
                    ).abs().mean((-2, -1))
            if "mss" in self.names:
                values["mss"] = ddsp_linear + ddsp_log

        if "smooth_mss" in self.names:
            smooth = rows.new_zeros(len(rows))
            for size, hop, window, saved in zip(
                SMOOTH_WINDOWS,
                SMOOTH_HOPS,
                self.smooth_windows,
                self.smooth_targets,
                strict=True,
            ):
                magnitude = _stft(
                    rows, n_fft=size, hop=hop, window=window, center=True
                ).abs()
                reference = self._select(saved, indices)
                smooth = smooth + (torch.log1p(magnitude) - reference).square().mean((-2, -1))
            values["smooth_mss"] = smooth / len(SMOOTH_WINDOWS)

        if any(name.startswith("sot_") for name in self.names):
            candidate_power = _stft(
                rows,
                n_fft=SOT_N_FFT,
                hop=SOT_HOP,
                window=self.sot_window,
                center=True,
            ).abs().square()
            target_power = self._select(self.sot_target_power, indices)
            x = candidate_power.transpose(-2, -1)
            y = target_power.transpose(-2, -1).expand_as(x)
            x_mass = x.sum(-1, keepdim=True) + 1e-8
            for name, cutoff in (
                ("sot_published_composite", True),
                ("sot_no_cut", False),
            ):
                if name not in self.names:
                    continue
                x_weights = x / x_mass
                y_weights = y / x_mass if cutoff else y / (y.sum(-1, keepdim=True) + 1e-8)
                transport = _quantile_wasserstein(
                    x_weights.reshape(-1, x.shape[-1]),
                    y_weights.reshape(-1, y.shape[-1]),
                    self.sot_positions,
                    limit_quantile_range=cutoff,
                ).reshape(x.shape[:2]).mean(-1)
                values[name] = transport + 0.05 * ddsp_linear

        if "linear_jtfot" in self.names or "log_jtfot" in self.names:
            magnitude = _stft(
                rows,
                n_fft=TFW_N_FFT,
                hop=TFW_HOP,
                window=self.tfw_window,
                center=True,
            ).abs()
            reference = self._select(self.tfw_target, indices)
            x = magnitude.flatten(1)
            y = reference.flatten(1).expand_as(x)
            for name in ("linear_jtfot", "log_jtfot"):
                if name not in self.names:
                    continue
                order = getattr(self, f"{name}_order")
                positions = getattr(self, f"{name}_positions")
                values[name] = _projected_wasserstein(
                    x[:, order], y[:, order], positions
                ).mean(-1)
        return values

    def __call__(self, candidate: Tensor, indices: Tensor | None = None) -> Tensor:
        if len(self.names) != 1:
            raise ValueError("direct calls require exactly one selected objective")
        return self.values(candidate, indices)[self.names[0]]


# Compatibility exports for the archived exploratory scripts. The frozen paper
# experiments above do not depend on these implementations.
from .legacy_losses import (  # noqa: E402, F401
    CEL_DIRECTIONS,
    CEL_NAMES,
    LOSS_LABELS,
    LOSS_NAMES,
    PAPER_LOSSES,
    BidirectionalCumulativeEnergyDistance,
    CumulativeEnergyDistance,
    LinearJTFOTDistance,
    LogJTFOTDistance,
    LogQuadratureBiCumulativeEnergyDistance,
    PublishedSOTCompositeDistance,
    SmoothMSSDistance,
    WaveformDistance,
    _linear_projection_geometry,
    _projection_geometry,
    _wasserstein_frequency_rows,
    _wasserstein_projected,
    build_loss,
    canonical_loss_name,
    periodic_flattop,
    reverse_cumsum,
)

SOT_MSS_WINDOWS: Final = DDSP_FFT_SIZES
SOT_MSS_HOPS: Final = DDSP_HOPS
FABIANI_TIME_SCALE_HZ_PER_SECOND: Final = TFW_HZ_PER_SECOND

__all__ = [
    "CEL_HOP",
    "CEL_N_FFT",
    "DDSP_FFT_SIZES",
    "DDSP_HOPS",
    "LABELS",
    "NAMES",
    "PaperObjectives",
    "SMOOTH_HOPS",
    "SMOOTH_WINDOWS",
    "SOT_HOP",
    "SOT_N_FFT",
    "TFW_HOP",
    "TFW_N_FFT",
    "CEL_DIRECTIONS",
    "CEL_NAMES",
    "LOSS_LABELS",
    "LOSS_NAMES",
    "PAPER_LOSSES",
    "SOT_MSS_HOPS",
    "SOT_MSS_WINDOWS",
    "BidirectionalCumulativeEnergyDistance",
    "CumulativeEnergyDistance",
    "LinearJTFOTDistance",
    "LogJTFOTDistance",
    "LogQuadratureBiCumulativeEnergyDistance",
    "PublishedSOTCompositeDistance",
    "SmoothMSSDistance",
    "WaveformDistance",
    "build_loss",
    "canonical_loss_name",
    "periodic_flattop",
    "reverse_cumsum",
]
