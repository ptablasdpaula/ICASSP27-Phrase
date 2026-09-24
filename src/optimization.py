"""Independent Adam fits with plateau scheduling and lowest-loss retention."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import Tensor

from .losses import LOSS_LABELS, PaperObjectives
from .synth import PhraseSynth
from .synth.config import EventPhrase, initial_candidate


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


@dataclass(frozen=True)
class FitResult:
    loss_name: str
    initial_loss: float
    best_loss: float
    best_phrase: EventPhrase
    trajectory: tuple[FitSnapshot, ...]
    updates: int
    evaluations: int
    plateau_events: int
    stopped_by: str
    wall_seconds: float


@dataclass(frozen=True)
class Schedule:
    initial_lr: float = 0.05
    factor: float = 0.5
    plateau_patience: int = 200
    threshold: float = 1e-4
    stop_patience: int = 1000
    maximum_updates: int = 20_000
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8

    def __post_init__(self):
        if not (self.initial_lr > 0 and 0 < self.factor < 1 and 0 <= self.threshold < 1):
            raise ValueError("Invalid learning rate, factor or relative threshold")
        if self.plateau_patience < 0 or self.stop_patience < 1 or self.maximum_updates < 1:
            raise ValueError("Invalid patience or update count")
        if not all(0 <= beta < 1 for beta in self.betas) or self.epsilon <= 0:
            raise ValueError("Invalid Adam parameters")


SCHEDULE = Schedule()


PLATEAU_EPSILON = 1e-8


def encode(f0: torch.Tensor, onset: torch.Tensor) -> torch.Tensor:
    f = (torch.log2(f0 / 80.0) / 2.0).clamp(1e-12, 1 - 1e-12)
    t = ((onset - 0.2) / 1.6).clamp(1e-12, 1 - 1e-12)
    return torch.stack((torch.logit(f), torch.logit(t)), -1)


def decode(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    unit = raw.sigmoid()
    return 80.0 * 4.0 ** unit[..., 0], 0.2 + 1.6 * unit[..., 1]


class PairedObjective(PaperObjectives):
    """One frozen paper objective for paired target/candidate rows."""

    def __init__(self, target: torch.Tensor, name: str):
        super().__init__(target, (name,))
        self.name = name

    def __call__(self, audio: torch.Tensor, indices: torch.Tensor | None = None) -> torch.Tensor:
        return self.values(audio, indices)[self.name]


def append_lr_events(events, indices, old_lr, lr, updates, strict_best):
    """Record compacted scheduler changes against their original batch rows."""
    for previous, i in zip(old_lr.tolist(), indices.tolist(), strict=True):
        events[i].append(
            {
                "update": int(updates[i]),
                "old_lr": previous,
                "new_lr": float(lr[i]),
                "best_loss": float(strict_best[i]),
            }
        )


def retain_strict_best(strict_best, best_raw, indices, values, selected_raw):
    """Retain each active phrase's parameters at its strict lowest loss."""
    detached = values.detach()
    strict = detached < strict_best[indices]
    improved_indices = indices[strict]
    strict_best[improved_indices] = detached[strict]
    best_raw[improved_indices] = selected_raw[strict].detach()
    return strict


def fit_batch(target_audio, cardinality, name, synth, *, schedule=SCHEDULE, progress=None):
    if target_audio.ndim != 2 or not bool(torch.isfinite(target_audio).all()):
        raise ValueError("target audio must be finite [batch,samples]")
    batch = len(target_audio)
    initial = initial_candidate(cardinality, device=target_audio.device)
    f0 = initial.f0_hz.expand(batch, -1)
    onset = initial.onset_seconds.expand(batch, -1)
    raw = encode(f0, onset).clone().requires_grad_(True)
    best_raw = raw.detach().clone()
    first, second = torch.zeros_like(raw), torch.zeros_like(raw)
    updates = torch.zeros(batch, dtype=torch.long, device=raw.device)
    lr = torch.full((batch,), schedule.initial_lr, dtype=torch.float64, device=raw.device)
    plateau_bad = torch.zeros(batch, dtype=torch.long, device=raw.device)
    stop_bad = torch.zeros(batch, dtype=torch.long, device=raw.device)
    reductions = torch.zeros(batch, dtype=torch.long, device=raw.device)
    active = torch.ones(batch, dtype=torch.bool, device=raw.device)
    initial_loss = scheduler_best = strict_best = final_loss = None
    events = [[] for _ in range(batch)]
    objective = PairedObjective(target_audio, name)
    beta1, beta2 = schedule.betas
    active_count = batch
    while active_count:
        indices = active.nonzero().flatten()
        selected_raw = raw[indices]
        f0, onset = decode(selected_raw)
        candidate_audio = synth(f0, onset)
        values = objective(candidate_audio, indices)
        if initial_loss is None:
            initial_loss = values.detach().clone()
            scheduler_best = values.detach().clone()
            strict_best = values.detach().clone()
            final_loss = values.detach().clone()
            keep = torch.ones_like(values, dtype=torch.bool)
        else:
            detached = values.detach()
            final_loss[indices] = detached
            retain_strict_best(strict_best, best_raw, indices, detached, selected_raw)
            meaningful = detached < scheduler_best[indices] * (1 - schedule.threshold)
            scheduler_best[indices] = torch.where(meaningful, detached, scheduler_best[indices])
            plateau_bad[indices] = torch.where(
                meaningful, torch.zeros_like(plateau_bad[indices]), plateau_bad[indices] + 1
            )
            stop_bad[indices] = torch.where(
                meaningful, torch.zeros_like(stop_bad[indices]), stop_bad[indices] + 1
            )
            # Match torch.optim.lr_scheduler.ReduceLROnPlateau: reduction
            # occurs when num_bad_epochs > patience, not at equality.
            due = plateau_bad[indices] > schedule.plateau_patience
            if bool(due.any()):
                due_indices = indices[due]
                old = lr[due_indices].clone()
                proposed = old * schedule.factor
                changed = old - proposed > PLATEAU_EPSILON
                reduced_indices = due_indices[changed]
                if len(reduced_indices):
                    lr[reduced_indices] = proposed[changed]
                    reductions[reduced_indices] += 1
                    append_lr_events(
                        events, reduced_indices, old[changed], lr, updates, strict_best
                    )
                # ReduceLROnPlateau resets only its own bad-epoch count.  The
                # independent early-stopping count deliberately continues.
                plateau_bad[due_indices] = 0
            stop = (stop_bad[indices] >= schedule.stop_patience) | (
                updates[indices] >= schedule.maximum_updates
            )
            keep = ~stop
            stopped_indices = indices[stop]
            active[stopped_indices] = False
            active_count -= len(stopped_indices)
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError("non-finite optimisation loss")
        if progress is not None and batch == 1:
            sf, st = decode(selected_raw.detach())
            snapshot = FitSnapshot(
                int(updates[0]) + 1,
                int(updates[0]),
                float(values[0].detach()),
                float(strict_best[0]),
                float(lr[0]),
                int(stop_bad[0]),
                int(reductions[0]),
                tuple(sf[0].tolist()),
                tuple(st[0].tolist()),
            )
            with torch.no_grad():
                progress(snapshot, candidate_audio[0].detach())
        if not bool(keep.any()):
            continue
        step_indices = indices[keep]
        scaled = (values[keep] / initial_loss[step_indices]).sum()
        (all_selected_gradient,) = torch.autograd.grad(scaled, selected_raw)
        gradient = all_selected_gradient[keep]
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("non-finite optimisation gradient")
        first[step_indices] = beta1 * first[step_indices] + (1 - beta1) * gradient
        second[step_indices] = beta2 * second[step_indices] + (1 - beta2) * gradient.square()
        updates[step_indices] += 1
        step = updates[step_indices].to(torch.float64)[:, None, None]
        delta = (
            lr[step_indices, None, None]
            * (first[step_indices] / (1 - beta1**step))
            / (torch.sqrt(second[step_indices] / (1 - beta2**step)) + schedule.epsilon)
        )
        next_raw = raw.detach().clone()
        next_raw[step_indices] -= delta
        raw = next_raw.requires_grad_(True)
    f0, onset = decode(best_raw)
    return (
        f0.detach(),
        onset.detach(),
        final_loss,
        strict_best,
        initial_loss,
        updates,
        reductions,
        events,
    )


def fit(
    target_audio: Tensor,
    cardinality: int,
    loss_name: str,
    *,
    synth=None,
    config: Schedule = SCHEDULE,
    progress=None,
) -> FitResult:
    """Fit one phrase using the campaign optimiser.

    ``progress(snapshot, audio)`` runs after every candidate evaluation,
    including the initial and terminal candidates. ``snapshot.update`` counts
    completed Adam steps; ``snapshot.evaluation`` is one greater. Audio is the
    detached candidate already rendered for the loss, with no extra synthesis.
    """
    renderer = synth or PhraseSynth().to(target_audio.device)
    name = LOSS_LABELS.get(loss_name, loss_name)
    if target_audio.ndim != 1 or target_audio.dtype != torch.float64:
        raise ValueError("target_audio must be a float64 sample vector")
    started = time.perf_counter()
    f, t, terminal, best, initial, updates, reductions, events = fit_batch(
        target_audio[None], cardinality, name, renderer, schedule=config, progress=progress
    )
    return FitResult(
        name,
        float(initial[0]),
        float(best[0]),
        EventPhrase(f[0], t[0]),
        (),
        int(updates[0]),
        int(updates[0]) + 1,
        int(reductions[0]),
        "maximum_updates" if int(updates[0]) >= config.maximum_updates else "patience",
        time.perf_counter() - started,
    )
