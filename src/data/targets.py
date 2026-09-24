"""Frozen held-out LHS target registry; audio is always rendered on demand."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any

import torch

from ..synth.config import CARDINALITIES, EventPhrase


@dataclass(frozen=True)
class Target:
    target_id: str
    index: int
    cardinality: int
    f0_hz: tuple[float, ...]
    onset_seconds: tuple[float, ...]

    def phrase(self, *, device: torch.device | str = "cpu") -> EventPhrase:
        return EventPhrase(
            torch.tensor(self.f0_hz, dtype=torch.float64, device=device),
            torch.tensor(self.onset_seconds, dtype=torch.float64, device=device),
        )


@lru_cache(maxsize=1)
def _registry() -> dict[tuple[int, int], Target]:
    path = files("icassp27_phrase").joinpath("data/targets.json")
    value: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "icassp27-phrase-targets-v2":
        raise ValueError("target registry has an unknown schema")
    if value.get("seed") != 2029 or tuple(value.get("cardinalities", ())) != CARDINALITIES:
        raise ValueError("target registry changed its registered design")
    targets: dict[tuple[int, int], Target] = {}
    counts = {cardinality: 0 for cardinality in CARDINALITIES}
    for record in value.get("targets", ()):
        cardinality = int(record["cardinality"])
        index = int(record["target_index"])
        events = record["events"]
        if cardinality not in CARDINALITIES or not 0 <= index < 150:
            raise ValueError("target registry contains an invalid index")
        if len(events) != cardinality:
            raise ValueError("target cardinality disagrees with its events")
        expected_id = f"C{cardinality:02d}-T{index:04d}"
        if record.get("target_id") != expected_id:
            raise ValueError("target registry contains a malformed ID")
        f0 = tuple(float(event["f0_hz"]) for event in events)
        onset = tuple(float(event["onset_seconds"]) for event in events)
        if not all(math.isfinite(item) for item in (*f0, *onset)):
            raise FloatingPointError("target registry contains non-finite coordinates")
        if any(not 80.0 <= item <= 320.0 for item in f0):
            raise ValueError("target pitch lies outside 80--320 Hz")
        if any(not 0.2 <= item <= 1.8 for item in onset):
            raise ValueError("target onset lies outside 0.2--1.8 s")
        if any(right - left < 0.05 - 1e-12 for left, right in zip(onset, onset[1:], strict=False)):
            raise ValueError("target onsets violate the 50-ms separation")
        key = (cardinality, index)
        if key in targets:
            raise ValueError("target registry contains a duplicate")
        targets[key] = Target(expected_id, index + 1, cardinality, f0, onset)
        counts[cardinality] += 1
    if counts != {cardinality: 150 for cardinality in CARDINALITIES}:
        raise ValueError("target registry must contain 150 phrases per cardinality")
    return targets


def load_target(
    number_of_events: int,
    target: int,
    *,
    device: torch.device | str = "cpu",
) -> tuple[Target, EventPhrase]:
    """Load target ``1..150`` for one registered event cardinality."""
    if number_of_events not in CARDINALITIES:
        raise ValueError("number_of_events must be one of 1, 2, 4, 6, or 8")
    if not 1 <= target <= 150:
        raise ValueError("target must be between 1 and 150 inclusive")
    metadata = _registry()[(number_of_events, target - 1)]
    return metadata, metadata.phrase(device=device)


__all__ = ["Target", "load_target"]
