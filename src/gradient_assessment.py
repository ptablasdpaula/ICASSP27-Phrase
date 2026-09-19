"""Paired LHS gradient assessment, with one second equal to one octave.

This is an isolated experiment; historical loss and recovery registries are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import qmc

from .losses import (
    SMOOTH_HOPS,
    SMOOTH_WINDOWS,
    SOT_MSS_HOPS,
    SOT_MSS_WINDOWS,
    CumulativeEnergyDistance,
    LinearJTFOTDistance,
    PublishedSOTCompositeDistance,
    SmoothMSSDistance,
    _linear_projection_geometry,
    _projection_geometry,
    _stft,
    _wasserstein_frequency_rows,
    _wasserstein_projected,
    reverse_cumsum,
)

SCHEMA = "gradient-assessment-v2"
SEED = 2028
NAMES = (
    "waveform_l1",
    "waveform_mse",
    "single_stft",
    "linear_mss",
    "smooth_mss",
    "sot_published_composite",
    "linear_jtfot",
    "log_jtfot",
    "bidirectional_cumulative_energy",
    "log_quadrature_bicul",
    "fading",
)
COLUMNS = (
    (1, "joint"),
    (2, "pitch"),
    (4, "pitch"),
    (2, "time"),
    (4, "time"),
    (2, "joint"),
    (4, "joint"),
)


def seed_for(*items):
    return int.from_bytes(hashlib.sha256(str((SEED, *items)).encode()).digest()[:4], "little")


def targets():
    """Coordinates are (log2(f/80), onset seconds); target rhythm is random."""
    result = [("C01-T000", np.array([[1.0, 1.0]]))]
    for n in (2, 4):
        for i in range(32):
            name = f"C{n:02d}-T{i:03d}"
            rng = np.random.default_rng(seed_for("target", name))
            pitch = rng.uniform(0, 2, n)
            while True:
                time = rng.uniform(0.2, 1.8, n)
                order = np.argsort(time)
                if np.min(np.diff(time[order])) >= 0.05:
                    break
            result.append((name, np.stack((pitch[order], time[order]), axis=-1)))
    return result


def candidates(name, target, count=256, repeat=0):
    unit = (
        qmc.LatinHypercube(
            d=target.size,
            scramble=True,
            optimization=None,
            seed=seed_for("candidate", name, count, repeat),
        )
        .random(count)
        .reshape(count, len(target), 2)
    )
    joint = unit * np.array([2.0, 1.6]) + np.array([0.0, 0.2])
    if len(target) == 1:
        return {"joint": joint}
    pitch, time = joint.copy(), joint.copy()
    pitch[..., 1] = target[:, 1]
    time[..., 0] = target[:, 0]
    return {"pitch": pitch, "time": time, "joint": joint}


def matching(candidate, target):
    """Unrestricted matching, squared octaves plus squared seconds."""
    cost = np.square(candidate[:, None] - target[None]).sum(-1)
    rows, cols = linear_sum_assignment(cost)
    best = cost[rows, cols].sum()
    tied = False
    if len(candidate) > 1:
        for row, col in zip(rows, cols, strict=True):
            alternative = cost.copy()
            alternative[row, col] = np.inf
            a, b = linear_sum_assignment(alternative)
            if alternative[a, b].sum() - best <= 1e-12:
                tied = True
                break
    return cols, tied


def score(gradients, positions, target, assignments, ties):
    """Return per-candidate/per-loss scores; exclude ties and exact matches."""
    if not np.isfinite(gradients).all():
        raise FloatingPointError("nonfinite gradients must be investigated")
    delta = (target[assignments] - positions)[:, None]
    active = np.abs(delta) > 1e-12
    descent = -gradients
    correct = descent * delta > 0
    event_active = active.any(-1)

    def average(value, mask):
        mask = np.broadcast_to(mask, value.shape)
        denominator = mask.sum(-1)
        return np.divide(
            np.where(mask, value, 0).sum(-1),
            denominator,
            out=np.full(denominator.shape, np.nan),
            where=denominator > 0,
        )

    success = (correct | ~active).all(-1)
    dot = (descent * delta).sum(-1)
    norms = np.linalg.norm(descent, axis=-1) * np.linalg.norm(delta, axis=-1)
    cosine = np.divide(dot, norms, out=np.zeros_like(dot), where=norms > 0)
    out = {
        "both_directed": average(success, event_active),
        "pitch_directed": average(correct[..., 0], active[..., 0]),
        "time_directed": average(correct[..., 1], active[..., 1]),
        "dot_directed": average(dot > 0, event_active),
        "cosine": average(cosine, event_active),
        "phrase_success": (success | ~event_active).all(-1).astype(float),
        "zero_gradient": average(np.linalg.norm(descent, axis=-1) == 0, event_active),
    }
    norm = np.linalg.norm(gradients.reshape(*gradients.shape[:2], -1), axis=-1)
    originally_correct = np.abs(positions - target)[:, None] <= 1e-12
    drift = np.linalg.norm(
        np.where(originally_correct, gradients, 0).reshape(*gradients.shape[:2], -1), axis=-1
    )
    out["original_coordinate_drift"] = np.divide(
        drift, norm, out=np.zeros_like(norm), where=norm > 0
    )
    out["original_coordinate_drift"] = np.where(
        originally_correct.any(axis=(-1, -2)), out["original_coordinate_drift"], np.nan
    )
    for value in out.values():
        value[ties | ~event_active[:, 0].any(-1)] = np.nan
    return out


def fade_matrix(size, spacing, horizon):
    index = torch.arange(size, dtype=torch.float64)
    distance = (index[:, None] - index[None, :]) * spacing
    return (1 - torch.log1p(9 * distance.clamp_min(0) / horizon) / math.log(10)).clamp_min(0) * (
        distance >= 0
    )


class SharedObjectives:
    """Cache target features; compute every identical candidate STFT only once."""

    def __init__(self, target):
        self.target = target.detach()
        self.cel = CumulativeEnergyDistance(target)
        self.smooth = SmoothMSSDistance(target)
        self.sot = PublishedSOTCompositeDistance(target)
        self.tfw = LinearJTFOTDistance(target)
        self.single_reference = self.cel._power(target[None]).sqrt()
        self.fade_frequency = fade_matrix(self.single_reference.shape[-2], 4000 / 256, 1000).to(
            target.device
        )
        self.fade_time = fade_matrix(self.single_reference.shape[-1], 64 / 4000, 1).to(
            target.device
        )
        self.fade_mass = self.cel._power(target[None]).sum()
        self.fade_reference = self.fade_features(self.cel._power(target[None])).detach()

    def fade_features(self, power):
        result = []
        for tr in (False, True):
            for fr in (False, True):
                axes = tuple(a for a, r in ((-2, fr), (-1, tr)) if r)
                oriented = power.flip(axes) if axes else power
                value = self.fade_frequency @ oriented @ self.fade_time.T
                value = value.flip(axes) if axes else value
                result.append((value / self.fade_mass).clamp_min(1e-12).sqrt())
        return torch.stack(result, dim=1)  # batch, direction, frequency, time

    def values(self, audio):
        difference = audio - self.target
        values = {
            "waveform_l1": difference.abs().mean(-1),
            "waveform_mse": difference.square().mean(-1),
        }
        magnitude = _stft(
            audio, n_fft=256, hop=64, window=self.cel.window, center=False, pad_mode="constant"
        ).abs()
        values["single_stft"] = (magnitude - self.single_reference).abs().mean((-2, -1))
        power = magnitude.square()
        ordinary, weighted = [], []
        for tr in (False, True):
            time_surface = reverse_cumsum(power, 2) if tr else power.cumsum(2)
            for fr in (False, True):
                surface = reverse_cumsum(time_surface, 1) if fr else time_surface.cumsum(1)
                index = len(ordinary)
                reference, mass = self.cel.references[index]
                error = (surface / mass).clamp_min(1e-12).sqrt() - reference
                ordinary.append(
                    torch.linalg.vector_norm(error.flatten(1), dim=1)
                    / math.sqrt(error.shape[-1] * error.shape[-2])
                )
                weighted.append(
                    torch.linalg.vector_norm(
                        (error * self.cel.sqrt_weights[index]).flatten(1), dim=1
                    )
                )
        values["bidirectional_cumulative_energy"] = torch.stack(ordinary, -1).mean(-1)
        values["log_quadrature_bicul"] = torch.stack(weighted, -1).mean(-1)
        error = self.fade_features(power) - self.fade_reference
        values["fading"] = (
            torch.linalg.vector_norm(error.flatten(2), dim=-1)
            / math.sqrt(error.shape[-1] * error.shape[-2])
        ).mean(-1)
        mss = audio.new_zeros(len(audio))
        for n, hop, window, reference in zip(
            SOT_MSS_WINDOWS, SOT_MSS_HOPS, self.sot.mss_windows, self.sot.mss_targets, strict=True
        ):
            mag = _stft(
                audio, n_fft=n, hop=hop, window=window, center=True, pad_mode="reflect"
            ).abs()
            mss = mss + (mag - reference).abs().mean((-2, -1))
        values["linear_mss"] = mss
        smooth = audio.new_zeros(len(audio))
        for n, hop, window, reference in zip(
            SMOOTH_WINDOWS, SMOOTH_HOPS, self.smooth.windows, self.smooth.references, strict=True
        ):
            mag = _stft(
                audio, n_fft=n, hop=hop, window=window, center=True, pad_mode="reflect"
            ).abs()
            smooth = smooth + (torch.log1p(mag) - reference).square().sum((-2, -1))
        values["smooth_mss"] = smooth
        sot_power = (
            _stft(
                audio,
                n_fft=512,
                hop=64,
                window=self.sot.sot_window,
                center=True,
                pad_mode="reflect",
            )
            .abs()
            .square()
        )
        x = sot_power.transpose(-2, -1)
        y = self.sot.target_power.expand(len(audio), -1, -1).transpose(-2, -1)
        mass = x.sum(-1, keepdim=True) + 1e-8
        transport = (
            _wasserstein_frequency_rows(
                (x / mass).reshape(-1, x.shape[-1]),
                (y / mass).reshape(-1, y.shape[-1]),
                self.sot.positions,
            )
            .reshape(x.shape[:2])
            .mean(-1)
        )
        # Differentiate only transport; add the cached Linear MSS gradient afterwards.
        values["sot_published_composite"] = transport
        mag = _stft(
            audio, n_fft=256, hop=128, window=self.tfw.window, center=True, pad_mode="constant"
        ).abs()
        reference = self.tfw.target_magnitude.expand(len(audio), -1, -1)
        mass_x, mass_y = mag.sum((-2, -1)), reference.sum((-2, -1))
        if bool((mass_x <= 1e-12).any()) or bool((mass_y <= 1e-12).any()):
            raise ValueError("transport is undefined on silence")
        x, y = mag.flatten(1) / mass_x[:, None], reference.flatten(1) / mass_y[:, None]
        for name, geometry in (
            ("linear_jtfot", _linear_projection_geometry),
            ("log_jtfot", _projection_geometry),
        ):
            order, positions = geometry(
                mag.shape[-2], mag.shape[-1], 4000, 256, 128, str(audio.device), "float64"
            )
            values[name] = _wasserstein_projected(x[:, order], y[:, order], positions).mean(-1)
        return values

    def evaluate(self, synth, positions):
        coordinates = torch.tensor(
            positions, dtype=torch.float64, device=self.target.device, requires_grad=True
        )
        audio = synth(80 * 2 ** coordinates[..., 0], coordinates[..., 1])
        terms = self.values(audio)
        values, gradients = [], []
        for i, name in enumerate(NAMES):
            (gradient,) = torch.autograd.grad(
                terms[name].sum(), coordinates, retain_graph=i < len(NAMES) - 1
            )
            values.append(terms[name].detach())
            gradients.append(gradient.detach())
        value = torch.stack(values, 1).cpu().numpy()
        gradient = torch.stack(gradients, 1).cpu().numpy()
        sot, mss = NAMES.index("sot_published_composite"), NAMES.index("linear_mss")
        value[:, sot] += 0.05 * value[:, mss]
        gradient[:, sot] += 0.05 * gradient[:, mss]
        if not np.isfinite(value).all() or not np.isfinite(gradient).all():
            raise FloatingPointError("nonfinite objective/gradient")
        return value, gradient


def signature():
    paths = [
        Path(__file__),
        *(
            Path(__file__).parent / name
            for name in (
                "losses.py",
                "synth.py",
                "exciter.py",
                "waveguide.py",
                "_fractional.py",
                "config.py",
                "runtime.py",
            )
        ),
    ]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), hashes
