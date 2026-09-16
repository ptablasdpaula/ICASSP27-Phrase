"""Deterministic, optimisation-free multi-event CeL gradient screening."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import qmc

from .config import CARDINALITIES
from .losses import CEL_NAMES, CumulativeEnergyDistance
from .synth import PhraseSynth

SEED = 2028
SCHEMA = "cel-gradient-screen-v1"
OFFSETS = np.array([-0.2, -0.1, -0.05, -0.025, 0.0, 0.025, 0.05, 0.1, 0.2])


@dataclass(frozen=True)
class ScreenTarget:
    name: str
    profile: str
    coordinates: np.ndarray  # [event, pitch/onset], both in [0,1]


def design() -> list[ScreenTarget]:
    """Independent targets; the final 150-target registry is never used."""
    rng = np.random.default_rng(SEED)
    targets = []
    for n in CARDINALITIES:
        times = np.linspace(0.4, 1.6, n) if n > 1 else np.array([1.0])
        pitches = {f"repeated_{p}": np.full(n, p) for p in (100, 160, 256)}
        if n > 1:
            pitches["ascending"] = np.geomspace(100, 256, n)
            pitches["descending"] = pitches["ascending"][::-1]
        for profile, f in pitches.items():
            coords = np.stack((np.log2(f / 80) / 2, (times - 0.2) / 1.6), -1)
            targets.append(ScreenTarget(f"C{n:02d}-{profile}", profile, coords))
        for i in range(30):
            while True:
                times = np.sort(rng.uniform(0.2, 1.8, n))
                if np.all(np.diff(times) >= 0.05):
                    break
            coords = np.stack((rng.uniform(0, 1, n), (times - 0.2) / 1.6), -1)
            targets.append(ScreenTarget(f"C{n:02d}-pilot-{i:02d}", "independent", coords))
    return targets


def candidates(target: ScreenTarget) -> tuple[np.ndarray, np.ndarray]:
    """Isolated conditional slices, full-range joint Sobol samples and fit start."""
    n = len(target.coordinates)
    rows, kinds = [], []
    if target.profile != "independent":
        for event in range(n):
            for dp in OFFSETS:
                for dt in OFFSETS:
                    if dp == dt == 0:
                        continue
                    row = target.coordinates.copy()
                    row[event] += (dp, dt)
                    if np.any((row < 0) | (row > 1)):
                        continue
                    rows.append(row)
                    kinds.append(
                        "isolated_timing"
                        if dp == 0
                        else "isolated_pitch"
                        if dt == 0
                        else "isolated_joint"
                    )
    seed = SEED + int.from_bytes(hashlib.sha256(target.name.encode()).digest()[:4], "little")
    samples = qmc.Sobol(2 * n, scramble=True, seed=seed).random_base2(5)
    rows.extend(samples.reshape(32, n, 2))
    kinds.extend(["simultaneous"] * 32)
    initial = np.stack((np.full(n, 0.5), (np.arange(n) + 0.5) / n), -1)
    rows.append(initial)
    kinds.append("initialisation")
    return np.stack(rows), np.array(kinds)


def matching(candidate: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, bool]:
    """Assignment and ties within absolute 1e-12 of total normalised cost."""
    cost = np.square(candidate[:, None] - target[None]).sum(-1)
    rows, columns = linear_sum_assignment(cost)
    best = cost[rows, columns].sum()
    tied = False
    if len(candidate) > 1:
        for i, j in zip(rows, columns, strict=True):
            alternative = cost.copy()
            alternative[i, j] = np.inf
            a, b = linear_sum_assignment(alternative)
            if alternative[a, b].sum() - best <= 1e-12:
                tied = True
                break
    return columns, tied


def subset_matrix() -> np.ndarray:
    matrix = np.zeros((30, 8))
    for row, name in enumerate(CEL_NAMES):
        mask = int(name.split("_")[1])
        indices = [i + (4 if name.endswith("_lw") else 0) for i in range(4) if mask & (1 << i)]
        matrix[row, indices] = 1 / len(indices)
    return matrix


def elementary(
    synth: PhraseSynth, objective: CumulativeEnergyDistance, positions: np.ndarray, device: str
) -> tuple[np.ndarray, np.ndarray]:
    coords = torch.tensor(positions, dtype=torch.float64, device=device, requires_grad=True)
    audio = synth(80 * 4 ** coords[..., 0], 0.2 + 1.6 * coords[..., 1])
    terms = objective.directional_distances(audio).flatten(1)
    gradients = []
    for index in range(8):
        (grad,) = torch.autograd.grad(terms[:, index].sum(), coords, retain_graph=index < 7)
        gradients.append(grad.detach().cpu().numpy())
    return terms.detach().cpu().numpy(), np.stack(gradients, 1)


def source_hash() -> str:
    digest = hashlib.sha256()
    for name in (
        "config.py",
        "losses.py",
        "cel_screen.py",
        "synth.py",
        "exciter.py",
        "waveguide.py",
        "_fractional.py",
        "runtime.py",
    ):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def compute_target(index: int, output: Path, device: str, chunk_size: int = 16) -> Path:
    target = design()[index]
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{target.name}.npz"
    signature = source_hash()
    if path.exists():
        with np.load(path) as data:
            if str(data["source_hash"]) != signature or str(data["device"]) != device:
                raise ValueError(f"stale or different-backend shard: {path}")
        return path
    synth = PhraseSynth().to(device)
    coords = torch.tensor(target.coordinates, dtype=torch.float64, device=device)
    with torch.no_grad():
        audio = synth(80 * 4 ** coords[None, :, 0], 0.2 + 1.6 * coords[None, :, 1])[0]
    objective = CumulativeEnergyDistance(audio)
    positions, kinds = candidates(target)
    values, gradients = [], []
    for start in range(0, len(positions), chunk_size):
        value, grad = elementary(synth, objective, positions[start : start + chunk_size], device)
        values.append(value)
        gradients.append(grad)
    assignments, ties = zip(*(matching(p, target.coordinates) for p in positions), strict=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        schema=SCHEMA,
        source_hash=signature,
        device=device,
        torch_version=torch.__version__,
        target=target.coordinates,
        candidates=positions,
        kinds=kinds,
        elementary_losses=np.concatenate(values),
        elementary_gradients=np.concatenate(gradients),
        assignments=np.stack(assignments),
        tied=np.array(ties),
        renderer_json=json.dumps(synth.provenance(), sort_keys=True),
    )
    temporary.replace(path)
    return path


def score(gradients: np.ndarray, displacement: np.ndarray) -> dict[str, np.ndarray]:
    """One score per variant/candidate, averaging events within that phrase.

    Gradients have shape [candidate,variant,event,2]. Correct-coordinate
    tolerance is 1e-12. Zero gradients fail wherever displacement is nonzero.
    """
    descent = -gradients
    delta = displacement[:, None]
    valid = np.isfinite(gradients).all(axis=(-1, -2))
    event_norm = np.linalg.norm(descent, axis=-1)
    delta_norm = np.linalg.norm(delta, axis=-1)
    active = delta_norm > 1e-12
    dot = (descent * delta).sum(-1)
    cosine = np.divide(
        dot, event_norm * delta_norm, out=np.zeros_like(dot), where=event_norm * delta_norm > 0
    )

    def average(value, mask):
        count = np.broadcast_to(mask, value.shape).sum(-1)
        return np.divide(
            np.where(mask, value, 0).sum(-1),
            count,
            out=np.full(count.shape, np.nan),
            where=count > 0,
        )

    result = {
        "joint_directed": average(dot > 0, active),
        "joint_cosine": average(cosine, active),
        "zero_gradient": average(event_norm == 0, active),
    }
    for axis, name in enumerate(("pitch", "onset")):
        result[f"{name}_directed"] = average(
            descent[..., axis] * delta[..., axis] > 0, np.abs(delta[..., axis]) > 1e-12
        )
    norm = np.linalg.norm(descent.reshape(*descent.shape[:2], -1), axis=-1)
    phrase_dot = dot.sum(-1)
    phrase_active = np.linalg.norm(displacement.reshape(len(displacement), -1), axis=-1) > 1e-12
    result["phrase_directed"] = np.where(phrase_active[:, None], phrase_dot > 0, np.nan)
    correct = np.abs(delta) <= 1e-12
    correct_norm = np.linalg.norm(
        np.where(correct, descent, 0).reshape(*descent.shape[:2], -1), axis=-1
    )
    result["correct_coordinate_drift"] = np.where(
        correct.any(axis=(-1, -2)),
        np.divide(correct_norm, norm, out=np.zeros_like(norm), where=norm > 0),
        np.nan,
    )
    result["nonfinite"] = ~valid
    for key in result:
        if key != "nonfinite":
            result[key] = np.where(valid, result[key], np.nan)
    return result
