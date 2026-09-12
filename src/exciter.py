"""Half-raised-cosine excitation and differentiable onset placement."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
import threading
import types
from pathlib import Path

import torch
from torch import Tensor, nn

from ._fractional import (
    causal_relocate_rows,
    hermitian_rfft_projection,
    lagrange_anchor,
    lagrange_weights,
    thiran_anchor,
    thiran_denominator,
    zero_state_filter_rows,
)
from .config import ExciterConfig

FLAMO_VERSION = "0.2.18"
_FLAMO_LOAD_LOCK = threading.RLock()
_PARALLEL_DELAY: type[nn.Module] | None = None


def half_raised_cosine(
    onset_seconds: Tensor,
    *,
    sample_rate: int,
    sample_count: int,
    amplitude: float,
    duration_seconds: float,
) -> Tensor:
    """Sample continuous HRC events; output shape is ``[batch,event,sample]``."""
    if onset_seconds.ndim != 2 or onset_seconds.dtype != torch.float64:
        raise TypeError("onsets must be a float64 [batch,event] tensor")
    sample = torch.arange(
        sample_count, dtype=torch.float64, device=onset_seconds.device
    )[None, None, :]
    offset = sample - onset_seconds[:, :, None] * float(sample_rate)
    width = float(duration_seconds) * float(sample_rate)
    support = ((offset >= 0.0) & (offset < width)).to(torch.float64)
    phase = torch.pi * offset / width
    return float(amplitude) * 0.5 * (1.0 - torch.cos(phase)) * support


def _parallel_delay_type() -> type[nn.Module]:
    """Load the official processor without FLAMO's unrelated optimizer stack."""
    global _PARALLEL_DELAY
    with _FLAMO_LOAD_LOCK:
        if _PARALLEL_DELAY is not None:
            return _PARALLEL_DELAY
        try:
            distribution = importlib.metadata.distribution("flamo")
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(
                "Fourier onset placement requires `flamo==0.2.18`; install the "
                "paper environment first"
            ) from error
        if distribution.version != FLAMO_VERSION:
            raise RuntimeError(
                f"expected FLAMO {FLAMO_VERSION}, found {distribution.version}"
            )
        package = Path(distribution.locate_file("flamo")).resolve()
        if not package.joinpath("processor", "dsp.py").is_file():
            raise RuntimeError("the installed FLAMO distribution has no processor source")

        existing = sys.modules.get("flamo")
        if existing is None:
            namespace = types.ModuleType("flamo")
            namespace.__path__ = [str(package)]
            namespace.__package__ = "flamo"
            sys.modules["flamo"] = namespace
        else:
            search_paths = getattr(existing, "__path__", None)
            if search_paths is None or package not in {
                Path(value).resolve() for value in search_paths
            }:
                raise RuntimeError("a different FLAMO package is already loaded")
        try:
            dsp = importlib.import_module("flamo.processor.dsp")
        except Exception as error:  # pragma: no cover - installation-specific
            raise RuntimeError("FLAMO parallelDelay could not be imported") from error
        source = Path(getattr(dsp, "__file__", "")).resolve()
        if source != package.joinpath("processor", "dsp.py"):
            raise RuntimeError("FLAMO processor was imported from an unexpected source")
        delay_type = getattr(dsp, "parallelDelay", None)
        if not isinstance(delay_type, type) or not issubclass(delay_type, nn.Module):
            raise RuntimeError("FLAMO 0.2.18 parallelDelay is unavailable")
        _PARALLEL_DELAY = delay_type
        return delay_type


class Exciter(nn.Module):
    """Generate and place one paper HRC prototype for every phrase event.

    ``naive`` samples the continuous HRC directly at each onset. ``lagrange``
    applies the registered fifth-order FIR delay, ``thiran`` applies a causal
    first-order all-pass delay, and ``fourier`` uses FLAMO's exact
    frequency-sampling delay response. Fourier event spectra are summed before
    the single inverse transform used by :meth:`forward`.
    """

    def __init__(self, config: ExciterConfig | None = None) -> None:
        super().__init__()
        self.config = config or ExciterConfig()
        self._delay_banks = nn.ModuleDict()
        self._bank_lock = threading.RLock()

    def prototype(self, *, device: torch.device | str) -> Tensor:
        """Return the zero-onset HRC prototype."""
        zero = torch.zeros((1, 1), dtype=torch.float64, device=device)
        return half_raised_cosine(
            zero,
            sample_rate=self.config.sample_rate,
            sample_count=self.config.sample_count,
            amplitude=self.config.amplitude,
            duration_seconds=self.config.duration_seconds,
        )[0, 0]

    def _validate(self, onset_seconds: Tensor) -> None:
        if onset_seconds.ndim != 2 or onset_seconds.dtype != torch.float64:
            raise TypeError("onsets must be a float64 [batch,event] tensor")
        if onset_seconds.shape[0] < 1 or onset_seconds.shape[1] < 1:
            raise ValueError("the exciter requires a non-empty batch and event axis")
        detached = onset_seconds.detach()
        duration = self.config.sample_count / self.config.sample_rate
        if not bool(torch.isfinite(detached).all()) or not bool(
            ((detached >= 0.0) & (detached < duration)).all()
        ):
            raise ValueError(f"onsets must be finite and lie in [0,{duration}) seconds")

    def _delay_bank(self, channels: int, device: torch.device) -> nn.Module:
        if channels < 1 or device.type not in {"cpu", "cuda"}:
            raise ValueError("FLAMO delays require positive channels on CPU or CUDA")
        index = device.index
        if device.type == "cuda" and index is None:
            index = torch.cuda.current_device()
        device_key = "cpu" if device.type == "cpu" else f"cuda_{index}"
        key = f"{device_key}_channels_{channels}"
        with self._bank_lock:
            if key in self._delay_banks:
                bank = self._delay_banks[key]
                if bank.param.device == device:
                    return bank
            delay_type = _parallel_delay_type()
            fork_devices = [index] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=fork_devices, enabled=True):
                bank = delay_type(
                    size=(channels,),
                    max_len=self.config.sample_count,
                    unit=1,
                    isint=False,
                    nfft=self.config.fourier_fft_length,
                    fs=self.config.sample_rate,
                    requires_grad=False,
                    alias_decay_db=0.0,
                    device=str(device),
                    dtype=torch.float64,
                )
            with torch.no_grad():
                bank.param.zero_()
            self._delay_banks[key] = bank
            return bank

    def _fourier_spectra(self, onset_seconds: Tensor) -> Tensor:
        batch, events = onset_seconds.shape
        flat = onset_seconds.reshape(-1)
        response = self._delay_bank(flat.numel(), onset_seconds.device).freq_response(flat)
        bins = self.config.fourier_fft_length // 2 + 1
        if response.shape != (bins, flat.numel()) or response.dtype != torch.complex128:
            raise RuntimeError("FLAMO returned an unexpected delay response")
        if not bool(torch.isfinite(response.detach()).all()):
            raise FloatingPointError("FLAMO returned a non-finite delay response")
        prototype_spectrum = torch.fft.rfft(
            self.prototype(device=onset_seconds.device),
            n=self.config.fourier_fft_length,
        )
        return response.T.reshape(batch, events, bins) * prototype_spectrum[None, None]

    def event_signals(self, onset_seconds: Tensor) -> Tensor:
        """Return individually placed events with shape ``[batch,event,sample]``."""
        self._validate(onset_seconds)
        method = self.config.method
        if method == "naive":
            result = half_raised_cosine(
                onset_seconds,
                sample_rate=self.config.sample_rate,
                sample_count=self.config.sample_count,
                amplitude=self.config.amplitude,
                duration_seconds=self.config.duration_seconds,
            )
        elif method == "fourier":
            result = torch.fft.irfft(
                hermitian_rfft_projection(
                    self._fourier_spectra(onset_seconds),
                    self.config.fourier_fft_length,
                ),
                n=self.config.fourier_fft_length,
                dim=-1,
            )[..., : self.config.sample_count]
        else:
            delay = onset_seconds.reshape(-1) * float(self.config.sample_rate)
            if method == "lagrange":
                order = self.config.lagrange_order
                whole = lagrange_anchor(delay, order)
                numerator = lagrange_weights(delay - whole.to(delay.dtype), order)
                denominator = delay.new_ones((delay.numel(), 1))
            elif method == "thiran":
                order = self.config.thiran_order
                whole = thiran_anchor(delay, order)
                denominator = thiran_denominator(delay, order, whole)
                numerator = torch.flip(denominator, dims=(-1,))
            else:  # pragma: no cover - ExciterConfig is exhaustive
                raise RuntimeError(f"unreachable onset method {method!r}")
            prototype = self.prototype(device=onset_seconds.device)
            source = prototype[None].expand(delay.numel(), -1)
            placed = causal_relocate_rows(
                zero_state_filter_rows(source, numerator, denominator),
                whole,
                self.config.sample_count,
            )
            result = placed.reshape(*onset_seconds.shape, self.config.sample_count)
        if result.dtype != torch.float64 or not bool(torch.isfinite(result.detach()).all()):
            raise FloatingPointError("the exciter produced malformed audio")
        if torch.is_grad_enabled() and onset_seconds.requires_grad and not result.requires_grad:
            raise RuntimeError("the onset-placement method detached its gradient")
        return result

    def forward(self, onset_seconds: Tensor) -> Tensor:
        """Place all events and sum them into one source signal per phrase."""
        self._validate(onset_seconds)
        if self.config.method == "fourier":
            spectrum = self._fourier_spectra(onset_seconds).sum(dim=1)
            result = torch.fft.irfft(
                hermitian_rfft_projection(spectrum, self.config.fourier_fft_length),
                n=self.config.fourier_fft_length,
                dim=-1,
            )[:, : self.config.sample_count]
        else:
            result = self.event_signals(onset_seconds).sum(dim=1)
        if result.shape != (onset_seconds.shape[0], self.config.sample_count):
            raise RuntimeError("the summed excitation has the wrong shape")
        return result


__all__ = ["Exciter", "FLAMO_VERSION", "half_raised_cosine"]
