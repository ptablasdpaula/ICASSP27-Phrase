"""Fixed LHS design for the paper's phrase-gradient assessment."""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from icassp27_phrase.losses import NAMES, PaperObjectives
from scipy.stats import qmc

from ..synth.controls import CONTROL_NAMES

TARGETS_PER_CARDINALITY = 32
CANDIDATES_PER_TARGET = 256
CARDINALITIES = (1, 2, 4, 6, 8)
COLUMNS = (
    (1, "joint"),
    (2, "joint"),
    (4, "joint"),
    (6, "joint"),
    (8, "joint"),
    (1, "pitch"),
    (2, "pitch"),
    (4, "pitch"),
    (1, "time"),
    (2, "time"),
    (4, "time"),
)
SEED = 2028
SCHEMA = "gradient-assessment-16k-fixed-lhs-v1"


def seed_for(*items: object) -> int:
    """Derive a stable SciPy-compatible seed in an experiment-specific namespace."""
    payload = (SEED, "fixed-lhs-v1", *items)
    return int.from_bytes(hashlib.sha256(str(payload).encode()).digest()[:4], "little")


def target_unit_design(events: int) -> np.ndarray:
    """Return the frozen 32 by 2N scrambled-LHS target design."""
    if events not in CARDINALITIES:
        raise ValueError(f"events must be one of {CARDINALITIES}")
    return (
        qmc.LatinHypercube(
            d=2 * events,
            scramble=True,
            optimization=None,
            seed=seed_for("targets", events),
        )
        .random(TARGETS_PER_CARDINALITY)
        .reshape(TARGETS_PER_CARDINALITY, events, 2)
    )


def targets() -> list[tuple[str, np.ndarray]]:
    """Generate independent LHS targets with 50-ms-separated ordered onsets.

    The latent onset coordinates are sorted within each phrase, compressed by
    the total reserved separation, then offset by temporal rank. This maps the
    full LHS without rejection while guaranteeing the registered bounds.
    """
    registry = []
    for events in CARDINALITIES:
        unit = target_unit_design(events)
        pitch = 2.0 * unit[..., 0]
        latent_onset = np.sort(unit[..., 1], axis=1)
        available = 1.6 - 0.05 * (events - 1)
        onset = 0.2 + available * latent_onset + 0.05 * np.arange(events)[None]
        coordinates = np.stack((pitch, onset), axis=-1)
        for index, target in enumerate(coordinates):
            registry.append((f"C{events:02d}-T{index:03d}", target))
    return registry


def candidates(name: str, target: np.ndarray) -> dict[str, np.ndarray]:
    """Return one paired 256-point LHS and its two conditional projections."""
    unit = (
        qmc.LatinHypercube(
            d=target.size,
            scramble=True,
            optimization=None,
            seed=seed_for("candidates", name),
        )
        .random(CANDIDATES_PER_TARGET)
        .reshape(CANDIDATES_PER_TARGET, len(target), 2)
    )
    joint = unit * np.array([2.0, 1.6]) + np.array([0.0, 0.2])
    pitch, time = joint.copy(), joint.copy()
    pitch[..., 1] = target[:, 1]
    time[..., 0] = target[:, 0]
    return {"joint": joint, "pitch": pitch, "time": time}


class FixedObjectives(PaperObjectives):
    """All frozen 16-kHz paper objectives."""

    def __init__(self, target: torch.Tensor) -> None:
        super().__init__(target, NAMES)

    def evaluate(self, synth, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coordinates = torch.tensor(
            positions, dtype=torch.float64, device=self.target.device, requires_grad=True
        )
        audio = synth(80 * 2 ** coordinates[..., 0], coordinates[..., 1])
        terms = self.values(audio)
        values, gradients = [], []
        for index, name in enumerate(NAMES):
            (gradient,) = torch.autograd.grad(
                terms[name].sum(), coordinates, retain_graph=index < len(NAMES) - 1
            )
            values.append(terms[name].detach())
            gradients.append(gradient.detach())
        value = torch.stack(values, 1).cpu().numpy()
        gradient = torch.stack(gradients, 1).cpu().numpy()
        if not np.isfinite(value).all() or not np.isfinite(gradient).all():
            raise FloatingPointError("nonfinite objective/gradient")
        return value, gradient


def signature():
    from ..paths import scientific_signature

    return scientific_signature()


def design_metadata() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "seed": SEED,
        "signature": signature()[0],
        "cardinalities": list(CARDINALITIES),
        "columns": [list(column) for column in COLUMNS],
        "targets_per_cardinality": TARGETS_PER_CARDINALITY,
        "candidates_per_target": CANDIDATES_PER_TARGET,
        "pairs_per_column": TARGETS_PER_CARDINALITY * CANDIDATES_PER_TARGET,
        "pairs_per_loss": len(COLUMNS) * TARGETS_PER_CARDINALITY * CANDIDATES_PER_TARGET,
        "target_design": "independent scrambled 2N-dimensional LHS per cardinality",
        "candidate_design": "independent scrambled 2N-dimensional LHS per target",
        "target_pitch_coordinate": "log2(f0/80) in [0,2]",
        "target_onset_transform": (
            "sort latent LHS onsets, scale to 1.6-(N-1)*0.05 seconds, "
            "then add 0.05 seconds per temporal rank"
        ),
        "candidate_spacing": "unrestricted",
        "matching": "unrestricted Hungarian: squared octaves + squared seconds",
        "losses": list(NAMES),
        "targets": {name: target.tolist() for name, target in targets()},
    }


__all__ = [
    "CANDIDATES_PER_TARGET",
    "CARDINALITIES",
    "COLUMNS",
    "FixedObjectives",
    "NAMES",
    "SCHEMA",
    "TARGETS_PER_CARDINALITY",
    "candidates",
    "design_metadata",
    "seed_for",
    "signature",
    "target_unit_design",
    "targets",
]


def persistent_targets() -> list[tuple[str, np.ndarray]]:
    return [(name, value) for name, value in targets() if len(value) == 2]


def all_control_targets() -> list[tuple[str, np.ndarray]]:
    design = qmc.LatinHypercube(
        d=len(CONTROL_NAMES),
        scramble=True,
        optimization=None,
        seed=seed_for("limitations", "all-control-targets"),
    ).random(TARGETS_PER_CARDINALITY)
    return [(f"A01-T{index:03d}", row) for index, row in enumerate(design)]


def all_control_candidates(name: str) -> np.ndarray:
    return qmc.LatinHypercube(
        d=len(CONTROL_NAMES),
        scramble=True,
        optimization=None,
        seed=seed_for("limitations", "all-control-candidates", name),
    ).random(CANDIDATES_PER_TARGET)
