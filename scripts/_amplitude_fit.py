"""Isolated amplitude pilot: registered Adam schedule plus optional log-amplitudes.

Copied from src/optimization.py at 9250d9b; production fitter is unchanged.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
from icassp27_phrase.config import PAPER_OPTIMIZER, EventPhrase, OptimizerConfig, initial_candidate
from icassp27_phrase.losses import build_loss, canonical_loss_name
from icassp27_phrase.synth import PhraseSynth
from torch import Tensor


def encode_coordinates(f0_hz: Tensor, onset_seconds: Tensor) -> Tensor:
    """Encode independent bounded pitch/onset controls as unconstrained logits."""
    f_unit = (torch.log(f0_hz / 80.0) / math.log(4.0)).clamp(1e-12, 1.0 - 1e-12)
    t_unit = ((onset_seconds - 0.2) / 1.6).clamp(1e-12, 1.0 - 1e-12)
    return torch.stack((torch.logit(f_unit), torch.logit(t_unit)), dim=-1)


def decode_coordinates(raw: Tensor) -> tuple[Tensor, Tensor]:
    """Decode logits without ordering, spacing, or canonicalisation."""
    if raw.ndim != 2 or raw.shape[-1] != 2 or raw.dtype != torch.float64:
        raise ValueError("raw coordinates must be float64 [event,2]")
    unit = torch.sigmoid(raw)
    f0_hz = 80.0 * torch.pow(4.0, unit[:, 0])
    onset_seconds = 0.2 + 1.6 * unit[:, 1]
    return f0_hz, onset_seconds


@dataclass(frozen=True)
class FitSnapshot:
    evaluation: int
    update: int
    raw_loss: float
    best_loss: float
    learning_rate: float
    patience: int
    plateau_events: int
    f0_hz: tuple[float, ...]
    onset_seconds: tuple[float, ...]
    amplitudes: tuple[float, ...]


@dataclass(frozen=True)
class FitResult:
    loss_name: str
    initial_loss: float
    best_loss: float
    best_phrase: EventPhrase
    best_amplitudes: tuple[float, ...]
    trajectory: tuple[FitSnapshot, ...]
    updates: int
    evaluations: int
    plateau_events: int
    stopped_by: str
    wall_seconds: float


ProgressCallback = Callable[[FitSnapshot, Tensor], None]


def fit(
    target_audio: Tensor,
    cardinality: int,
    loss_name: str,
    *,
    synth: PhraseSynth | None = None,
    config: OptimizerConfig = PAPER_OPTIMIZER,
    progress: ProgressCallback | None = None,
    initial: EventPhrase | None = None,
    free_amplitudes: bool = False,
    coordinate_encoder: Callable = encode_coordinates,
    coordinate_decoder: Callable = decode_coordinates,
) -> FitResult:
    """Reproduce one registered gradient-descent fit entirely in memory."""
    renderer = synth or PhraseSynth()
    if target_audio.ndim != 1 or target_audio.dtype != torch.float64:
        raise ValueError("target_audio must be a float64 sample vector")
    if target_audio.shape[0] != renderer.sample_count:
        raise ValueError("target_audio length differs from the renderer")
    if not bool(torch.isfinite(target_audio.detach()).all()):
        raise FloatingPointError("target audio contains a non-finite value")
    canonical_name = canonical_loss_name(loss_name)
    initial = initial or initial_candidate(cardinality, device=target_audio.device)
    raw = coordinate_encoder(initial.f0_hz, initial.onset_seconds)
    if free_amplitudes:
        raw = torch.cat((raw, torch.zeros_like(raw[:, :1])), dim=1)
    raw = raw.requires_grad_(True)
    first = torch.zeros_like(raw)
    second = torch.zeros_like(raw)
    update_count = 0
    learning_rate = config.learning_rate
    initial_loss: float | None = None
    best_loss: float | None = None
    best_raw: Tensor | None = None
    meaningful_reference: float | None = None
    patience = 0
    plateau_events = 0
    trajectory: list[FitSnapshot] = []
    objective = build_loss(canonical_name, target_audio)
    beta1, beta2 = config.betas
    started = time.perf_counter()
    stopped_by = "maximum updates"

    while True:
        f0_hz, onset_seconds = coordinate_decoder(raw[:, :2])
        amplitudes = 0.8 * raw[:, 2].exp() if free_amplitudes else torch.full_like(f0_hz, 0.8)
        audio = render_amplitudes(renderer, f0_hz, onset_seconds, amplitudes)
        value_tensor = objective(audio)
        if value_tensor.ndim != 0:
            raise RuntimeError("a single fit must produce a scalar loss")
        if not all(
            bool(torch.isfinite(value.detach()).all())
            for value in (raw, f0_hz, onset_seconds, audio, value_tensor)
        ):
            raise FloatingPointError("non-finite value at an optimisation boundary")
        value = float(value_tensor.detach())

        if initial_loss is None:
            if value <= 0.0:
                raise FloatingPointError("initial loss must be positive")
            initial_loss = value
            best_loss = value
            best_raw = raw.detach().clone()
            meaningful_reference = value
        else:
            patience += 1
            if value < float(best_loss):
                best_loss = value
                best_raw = raw.detach().clone()
            if value <= float(meaningful_reference) * (
                1.0 - config.meaningful_relative_improvement
            ):
                meaningful_reference = value
                patience = 0
                plateau_events = 0

        snapshot = FitSnapshot(
            evaluation=len(trajectory) + 1,
            update=update_count,
            raw_loss=value,
            best_loss=float(best_loss),
            learning_rate=learning_rate,
            patience=patience,
            plateau_events=plateau_events,
            f0_hz=tuple(float(item) for item in f0_hz.detach().cpu()),
            onset_seconds=tuple(float(item) for item in onset_seconds.detach().cpu()),
            amplitudes=tuple(float(item) for item in amplitudes.detach().cpu()),
        )
        trajectory.append(snapshot)
        if progress is not None:
            progress(snapshot, audio.detach())

        if patience >= config.stop_patience:
            stopped_by = "patience"
            break
        if update_count >= config.maximum_updates:
            stopped_by = "maximum updates"
            break
        if patience >= (plateau_events + 1) * config.plateau_patience:
            plateau_events += 1
            raw = best_raw.detach().clone().requires_grad_(True)
            first = torch.zeros_like(first)
            second = torch.zeros_like(second)
            learning_rate = max(learning_rate * config.lr_factor, config.minimum_learning_rate)
            continue

        conditioned = value_tensor / initial_loss
        (gradient,) = torch.autograd.grad(conditioned, raw)
        if not bool(torch.isfinite(gradient.detach()).all()):
            raise FloatingPointError("the optimiser produced a non-finite gradient")
        update_count += 1
        first = beta1 * first + (1.0 - beta1) * gradient
        second = beta2 * second + (1.0 - beta2) * gradient.square()
        first_hat = first / (1.0 - beta1**update_count)
        second_hat = second / (1.0 - beta2**update_count)
        raw = (
            (raw - learning_rate * first_hat / (torch.sqrt(second_hat) + config.epsilon))
            .detach()
            .requires_grad_(True)
        )

    if best_raw is None or initial_loss is None or best_loss is None:
        raise RuntimeError("the optimiser did not evaluate its initial state")
    best_f0, best_onset = coordinate_decoder(best_raw[:, :2])
    best_phrase = EventPhrase(best_f0.detach(), best_onset.detach())
    return FitResult(
        loss_name=canonical_name,
        initial_loss=initial_loss,
        best_loss=best_loss,
        best_phrase=best_phrase,
        best_amplitudes=tuple((0.8 * best_raw[:, 2].exp()).tolist())
        if free_amplitudes
        else (0.8,) * cardinality,
        trajectory=tuple(trajectory),
        updates=update_count,
        evaluations=len(trajectory),
        plateau_events=plateau_events,
        stopped_by=stopped_by,
        wall_seconds=time.perf_counter() - started,
    )


__all__ = [
    "FitResult",
    "FitSnapshot",
    "decode_coordinates",
    "encode_coordinates",
    "fit",
]


def render_amplitudes(renderer, f0, onset, amplitudes):
    """Scale individual Fourier excitation spectra before summing and filtering."""
    from icassp27_phrase._fractional import hermitian_rfft_projection

    order = onset.detach().argsort(stable=True)
    f0, onset, amplitudes = f0[order][None], onset[order][None], amplitudes[order][None]
    exciter = renderer.exciter
    spectrum = (
        exciter._fourier_spectra(onset) * (amplitudes / exciter.config.amplitude)[..., None]
    ).sum(dim=1)
    source = torch.fft.irfft(
        hermitian_rfft_projection(spectrum, exciter.config.fourier_fft_length),
        n=exciter.config.fourier_fft_length,
        dim=-1,
    )[..., : renderer.sample_count]
    return renderer.waveguide(source, f0, onset)[0]
