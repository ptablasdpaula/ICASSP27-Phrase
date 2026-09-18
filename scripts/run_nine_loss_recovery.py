"""Matched nine-loss phrase recovery with per-phrase relative plateaus.

Each array shard processes ten frozen targets for one loss/cardinality on CUDA.
The vectorised optimiser implements independent Adam and ReduceLROnPlateau state
for every phrase; inactive rows remain masked until the shard is complete.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from icassp27_phrase.config import CARDINALITIES, initial_candidate
from icassp27_phrase.gradient_assessment import fade_matrix
from icassp27_phrase.losses import (
    SMOOTH_HOPS,
    SMOOTH_WINDOWS,
    SOT_MSS_HOPS,
    SOT_MSS_WINDOWS,
    CumulativeEnergyDistance,
    _linear_projection_geometry,
    _projection_geometry,
    _stft,
    _wasserstein_projected,
    reverse_cumsum,
)
from icassp27_phrase.runtime import require_df2_backend
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.targets import load_target
from scipy.optimize import linear_sum_assignment

ROOT = Path("results/nine-loss-recovery")
LOSSES = (
    "single_stft",
    "linear_mss",
    "smooth_mss",
    "linear_jtfot",
    "log_jtfot",
    "cel",
    "log_cel",
    "dec_cel",
    "tlog_cel",
)
LABELS = ("SS", "LinMSS", "SmoMSS", "TFW2", "logTFW2", "CeL", "logCeL", "decCeL", "tlogCeL")
TARGETS_PER_SHARD = 10
SHARDS_PER_CELL = 150 // TARGETS_PER_SHARD
TOTAL_SHARDS = len(LOSSES) * len(CARDINALITIES) * SHARDS_PER_CELL


@dataclass(frozen=True)
class Schedule:
    initial_lr: float = 0.05
    factor: float = 0.5
    plateau_patience: int = 200
    threshold: float = 1e-4
    minimum_lr: float = 1e-5
    stop_patience: int = 1000
    maximum_updates: int = 20_000
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8


SCHEDULE = Schedule()


def encode(f0: torch.Tensor, onset: torch.Tensor) -> torch.Tensor:
    f = (torch.log2(f0 / 80.0) / 2.0).clamp(1e-12, 1 - 1e-12)
    t = ((onset - 0.2) / 1.6).clamp(1e-12, 1 - 1e-12)
    return torch.stack((torch.logit(f), torch.logit(t)), -1)


def decode(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    unit = raw.sigmoid()
    return 80.0 * 4.0 ** unit[..., 0], 0.2 + 1.6 * unit[..., 1]


class PairedObjective:
    """One loss for paired target/candidate rows."""

    def __init__(self, target: torch.Tensor, name: str):
        self.name = name
        self.target = target.detach()
        self.window256 = torch.hann_window(
            256, periodic=True, dtype=target.dtype, device=target.device
        )
        if name == "single_stft":
            self.reference = self._mag(target, 256, 64, self.window256, False)
        elif name == "linear_mss":
            self.windows = [
                torch.hann_window(n, periodic=True, dtype=target.dtype, device=target.device)
                for n in SOT_MSS_WINDOWS
            ]
            self.references = [
                self._mag(target, n, h, w, True, "reflect")
                for n, h, w in zip(SOT_MSS_WINDOWS, SOT_MSS_HOPS, self.windows, strict=True)
            ]
        elif name == "smooth_mss":
            import scipy.signal.windows

            self.windows = [
                torch.as_tensor(
                    scipy.signal.windows.flattop(n, sym=False),
                    dtype=target.dtype,
                    device=target.device,
                )
                for n in SMOOTH_WINDOWS
            ]
            self.references = [
                torch.log1p(self._mag(target, n, h, w, True, "reflect"))
                for n, h, w in zip(SMOOTH_WINDOWS, SMOOTH_HOPS, self.windows, strict=True)
            ]
        elif name in {"linear_jtfot", "log_jtfot"}:
            self.reference = self._mag(target, 256, 128, self.window256, True)
            geometry = (
                _linear_projection_geometry if name == "linear_jtfot" else _projection_geometry
            )
            self.order, self.positions = geometry(
                self.reference.shape[-2],
                self.reference.shape[-1],
                4000,
                256,
                128,
                str(target.device),
                "float64",
            )
        else:
            power = self._mag(target, 256, 64, self.window256, False).square()
            self.mass = power.sum((-2, -1)).clamp_min(torch.finfo(target.dtype).tiny)
            ordinary = name != "dec_cel"
            if ordinary:
                surfaces = self._cumulative(power)
            else:
                self.fade_f = fade_matrix(power.shape[-2], 4000 / 256, 1000).to(target.device)
                self.fade_t = fade_matrix(power.shape[-1], 64 / 4000, 1).to(target.device)
                surfaces = self._decayed(power)
            self.references = torch.sqrt(
                (surfaces / self.mass[:, None, None, None]).clamp_min(1e-12)
            )
            base = CumulativeEnergyDistance(target[0])
            self.sqrt_weights = base.sqrt_weights.to(target.device)

    @staticmethod
    def _mag(audio, n, hop, window, center, pad="constant"):
        return _stft(audio, n_fft=n, hop=hop, window=window, center=center, pad_mode=pad).abs()

    @staticmethod
    def _cumulative(power):
        rows = []
        for tr in (False, True):
            x = reverse_cumsum(power, 2) if tr else power.cumsum(2)
            for fr in (False, True):
                rows.append(reverse_cumsum(x, 1) if fr else x.cumsum(1))
        return torch.stack(rows, 1)

    def _decayed(self, power):
        rows = []
        for tr in (False, True):
            for fr in (False, True):
                axes = tuple(a for a, r in ((-2, fr), (-1, tr)) if r)
                x = power.flip(axes) if axes else power
                x = self.fade_f @ x @ self.fade_t.T
                rows.append(x.flip(axes) if axes else x)
        return torch.stack(rows, 1)

    def __call__(self, audio):
        name = self.name
        if name == "single_stft":
            return (
                (self._mag(audio, 256, 64, self.window256, False) - self.reference)
                .abs()
                .mean((-2, -1))
            )
        if name in {"linear_mss", "smooth_mss"}:
            out = audio.new_zeros(len(audio))
            ns, hs = (
                (SOT_MSS_WINDOWS, SOT_MSS_HOPS)
                if name == "linear_mss"
                else (SMOOTH_WINDOWS, SMOOTH_HOPS)
            )
            for n, h, w, ref in zip(ns, hs, self.windows, self.references, strict=True):
                mag = self._mag(audio, n, h, w, True, "reflect")
                out += (
                    (mag - ref).abs().mean((-2, -1))
                    if name == "linear_mss"
                    else (torch.log1p(mag) - ref).square().sum((-2, -1))
                )
            return out
        if name in {"linear_jtfot", "log_jtfot"}:
            mag = self._mag(audio, 256, 128, self.window256, True)
            x = mag.flatten(1) / mag.sum((-2, -1))[:, None]
            y = self.reference.flatten(1) / self.reference.sum((-2, -1))[:, None]
            return _wasserstein_projected(x[:, self.order], y[:, self.order], self.positions).mean(
                -1
            )
        power = self._mag(audio, 256, 64, self.window256, False).square()
        surfaces = self._decayed(power) if name == "dec_cel" else self._cumulative(power)
        error = (
            torch.sqrt((surfaces / self.mass[:, None, None, None]).clamp_min(1e-12))
            - self.references
        )
        if name in {"log_cel", "tlog_cel"}:
            terms = torch.linalg.vector_norm((error * self.sqrt_weights[None]).flatten(2), dim=-1)
        else:
            terms = torch.linalg.vector_norm(error.flatten(2), dim=-1) / math.sqrt(
                error.shape[-2] * error.shape[-1]
            )
        return terms[:, :2].mean(-1) if name == "tlog_cel" else terms.mean(-1)


def assignment_metrics(f0, onset, target_f0, target_onset):
    p = np.log2(np.asarray(f0) / 80.0)
    t = np.asarray(onset)
    tp = np.log2(np.asarray(target_f0) / 80.0)
    tt = np.asarray(target_onset)
    cost = (p[:, None] - tp[None]) ** 2 + (t[:, None] - tt[None]) ** 2
    _, assignment = linear_sum_assignment(cost)
    old_cost = ((p[:, None] - tp[None]) / 2) ** 2 + ((t[:, None] - tt[None]) / 1.6) ** 2
    _, old_assignment = linear_sum_assignment(old_cost)
    return {
        "pitch_mae_cents": float(np.abs(1200 * (p - tp[assignment])).mean()),
        "onset_mae_ms": float(np.abs(1000 * (t - tt[assignment])).mean()),
        "assignment": assignment.tolist(),
        "legacy_assignment": old_assignment.tolist(),
        "assignment_changed_from_legacy": bool(not np.array_equal(assignment, old_assignment)),
    }


def fit(target_audio, cardinality, name, synth):
    batch = len(target_audio)
    initial = initial_candidate(cardinality, device=target_audio.device)
    f0 = initial.f0_hz.expand(batch, -1)
    onset = initial.onset_seconds.expand(batch, -1)
    raw = encode(f0, onset).clone().requires_grad_(True)
    first, second = torch.zeros_like(raw), torch.zeros_like(raw)
    updates = torch.zeros(batch, dtype=torch.long, device=raw.device)
    lr = torch.full((batch,), SCHEDULE.initial_lr, dtype=torch.float64, device=raw.device)
    plateau_bad = torch.zeros(batch, dtype=torch.long, device=raw.device)
    stop_bad = torch.zeros(batch, dtype=torch.long, device=raw.device)
    reductions = torch.zeros(batch, dtype=torch.long, device=raw.device)
    active = torch.ones(batch, dtype=torch.bool, device=raw.device)
    initial_loss = scheduler_best = strict_best = None
    best_raw = raw.detach().clone()
    events = [[] for _ in range(batch)]
    objective = PairedObjective(target_audio, name)
    beta1, beta2 = SCHEDULE.betas
    while bool(active.any()):
        f0, onset = decode(raw)
        values = objective(synth(f0, onset))
        if initial_loss is None:
            initial_loss = values.detach().clone()
            scheduler_best = values.detach().clone()
            strict_best = values.detach().clone()
        else:
            detached = values.detach()
            strict = detached < strict_best
            strict_best = torch.where(strict, detached, strict_best)
            best_raw = torch.where(strict[:, None, None], raw.detach(), best_raw)
            meaningful = detached < scheduler_best * (1 - SCHEDULE.threshold)
            scheduler_best = torch.where(meaningful, detached, scheduler_best)
            plateau_bad = torch.where(
                active & meaningful, torch.zeros_like(plateau_bad), plateau_bad + active.long()
            )
            stop_bad = torch.where(
                active & meaningful, torch.zeros_like(stop_bad), stop_bad + active.long()
            )
            # Match torch.optim.lr_scheduler.ReduceLROnPlateau: reduction
            # occurs when num_bad_epochs > patience, not at equality.
            reducible = (
                active
                & (plateau_bad > SCHEDULE.plateau_patience)
                & (lr > SCHEDULE.minimum_lr + 1e-15)
            )
            if bool(reducible.any()):
                old = lr.clone()
                lr = torch.where(
                    reducible, torch.clamp(lr * SCHEDULE.factor, min=SCHEDULE.minimum_lr), lr
                )
                reductions += reducible.long()
                # ReduceLROnPlateau resets only its own bad-epoch count.  The
                # independent early-stopping count deliberately continues.
                plateau_bad = torch.where(reducible, torch.zeros_like(plateau_bad), plateau_bad)
                for i in reducible.nonzero().flatten().tolist():
                    events[i].append(
                        {
                            "update": int(updates[i]),
                            "old_lr": float(old[i]),
                            "new_lr": float(lr[i]),
                            "best_loss": float(strict_best[i]),
                        }
                    )
            stop = active & (stop_bad > SCHEDULE.stop_patience)
            stop |= active & (updates >= SCHEDULE.maximum_updates)
            active &= ~stop
        if not bool(active.any()):
            break
        scaled = torch.where(active, values / initial_loss, torch.zeros_like(values)).sum()
        (gradient,) = torch.autograd.grad(scaled, raw)
        mask = active[:, None, None]
        first = torch.where(mask, beta1 * first + (1 - beta1) * gradient, first)
        second = torch.where(mask, beta2 * second + (1 - beta2) * gradient.square(), second)
        updates += active.long()
        step = updates.clamp_min(1).to(torch.float64)[:, None, None]
        delta = (
            lr[:, None, None]
            * (first / (1 - beta1**step))
            / (torch.sqrt(second / (1 - beta2**step)) + SCHEDULE.epsilon)
        )
        raw = torch.where(mask, raw - delta, raw).detach().requires_grad_(True)
    f0, onset = decode(best_raw)
    return f0.detach(), onset.detach(), strict_best, initial_loss, updates, reductions, events


def signature():
    paths = [Path(__file__), *sorted(Path("src").glob("*.py")), Path("src/data/targets.json")]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), hashes


def shard_coordinates(shard):
    cell, block = divmod(shard, SHARDS_PER_CELL)
    loss_index, cardinality_index = divmod(cell, len(CARDINALITIES))
    return LOSSES[loss_index], CARDINALITIES[cardinality_index], block * TARGETS_PER_SHARD


def run(shard, device="cuda"):
    name, cardinality, begin = shard_coordinates(shard)
    sig, hashes = signature()
    outpath = (
        ROOT
        / "raw"
        / name
        / f"C{cardinality:02d}-T{begin:04d}-{begin + TARGETS_PER_SHARD - 1:04d}.json.gz"
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    if outpath.exists():
        with gzip.open(outpath, "rt") as f:
            old = json.load(f)
        if old["signature"] != sig:
            raise ValueError(f"stale shard {outpath}")
        print("EXISTS", outpath, flush=True)
        return
    synth = PhraseSynth().to(device)
    metadata, phrases = zip(
        *(
            load_target(cardinality, i + 1, device=device)
            for i in range(begin, begin + TARGETS_PER_SHARD)
        ),
        strict=True,
    )
    # All rows share cardinality, so render the ten frozen targets in one
    # backend call rather than paying ten Python/extension launch sequences.
    with torch.no_grad():
        target_audio = synth(
            torch.stack([p.f0_hz for p in phrases]),
            torch.stack([p.onset_seconds for p in phrases]),
        )
    started = time.perf_counter()
    f0, onset, best, initial, updates, reductions, lr_events = fit(
        target_audio, cardinality, name, synth
    )
    rows = []
    for i, meta in enumerate(metadata):
        metrics = assignment_metrics(
            f0[i].cpu().numpy(), onset[i].cpu().numpy(), meta.f0_hz, meta.onset_seconds
        )
        rows.append(
            {
                "target_id": meta.target_id,
                "target": asdict(meta),
                "best_f0_hz": f0[i].cpu().tolist(),
                "best_onset_seconds": onset[i].cpu().tolist(),
                "initial_loss": float(initial[i]),
                "best_loss": float(best[i]),
                "updates": int(updates[i]),
                "lr_reductions": int(reductions[i]),
                "lr_events": lr_events[i],
                "metrics": metrics,
            }
        )
    payload = {
        "schema": "nine-loss-recovery-v1",
        "signature": sig,
        "source_hashes": hashes,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "shard": shard,
        "loss": name,
        "label": LABELS[LOSSES.index(name)],
        "cardinality": cardinality,
        "target_range": [begin, begin + TARGETS_PER_SHARD],
        "schedule": asdict(SCHEDULE),
        "matching": "squared octaves plus squared seconds; one octave equals one second",
        "target_minimum_separation_seconds": 0.05,
        "renderer": synth.provenance(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "wall_seconds": time.perf_counter() - started,
        "rows": rows,
    }
    temporary = outpath.with_suffix(".tmp.gz")
    with gzip.open(temporary, "wt") as f:
        json.dump(payload, f, separators=(",", ":"), allow_nan=False)
    temporary.replace(outpath)
    print(
        "COMPLETE", shard, name, cardinality, begin, round(payload["wall_seconds"], 2), flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.shard < TOTAL_SHARDS:
        raise ValueError(f"shard must be in [0,{TOTAL_SHARDS})")
    torch.set_num_threads(1)
    require_df2_backend(args.device)
    run(args.shard, args.device)
