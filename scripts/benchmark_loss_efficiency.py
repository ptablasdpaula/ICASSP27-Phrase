#!/usr/bin/env python3
"""Benchmark the end-to-end cost of the four representative paper losses."""

from __future__ import annotations

import csv
import gc
import json
import math
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from icassp27_phrase.config import CARDINALITIES, initial_candidate
from icassp27_phrase.losses import PaperObjectives
from icassp27_phrase.runtime import require_df2_backend
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.targets import load_target


LOSSES = {
    "smooth_mss": "SmoMSS",
    "sot_published_composite": "SOT",
    "linear_jtfot": "TFW2",
    "cel": "CeL",
}


def encode(f0: torch.Tensor, onset: torch.Tensor) -> torch.Tensor:
    f = (torch.log2(f0 / 80.0) / 2.0).clamp(1e-12, 1.0 - 1e-12)
    t = ((onset - 0.2) / 1.6).clamp(1e-12, 1.0 - 1e-12)
    return torch.stack((torch.logit(f), torch.logit(t)), dim=-1)


def decode(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    unit = raw.sigmoid()
    return 80.0 * 4.0 ** unit[..., 0], 0.2 + 1.6 * unit[..., 1]


def recovery_updates(
    path: Path, cardinalities: tuple[int, ...]
) -> dict[tuple[str, int], float]:
    grouped: dict[tuple[str, int], list[int]] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["label"], int(row["cardinality"]))
            if row["label"] in LOSSES.values():
                grouped.setdefault(key, []).append(int(row["updates"]))
    expected = {(label, n) for label in LOSSES.values() for n in cardinalities}
    grouped = {key: values for key, values in grouped.items() if key in expected}
    if set(grouped) != expected or any(len(values) != 150 for values in grouped.values()):
        raise RuntimeError("the recovery archive is incomplete for the benchmark losses")
    return {key: statistics.median(values) for key, values in grouped.items()}


def benchmark(
    output: Path,
    *,
    batch: int,
    warmup: int,
    measured: int,
    repeats: int,
    cardinalities: tuple[int, ...],
) -> None:
    device = torch.device("cuda")
    require_df2_backend(device)
    torch.set_num_threads(1)
    synth = PhraseSynth().to(device)
    medians = recovery_updates(
        Path("docs/phrase-recovery/16k/per_phrase.csv"), cardinalities
    )
    rows: list[dict[str, object]] = []

    for cardinality in cardinalities:
        phrases = [load_target(cardinality, index + 1, device=device)[1] for index in range(batch)]
        with torch.no_grad():
            targets = synth(
                torch.stack([phrase.f0_hz for phrase in phrases]),
                torch.stack([phrase.onset_seconds for phrase in phrases]),
            )
        initial = initial_candidate(cardinality, device=device)

        for loss, label in LOSSES.items():
            objective = PaperObjectives(targets, (loss,))
            raw = encode(
                initial.f0_hz.expand(batch, -1),
                initial.onset_seconds.expand(batch, -1),
            ).clone().requires_grad_(True)
            torch.cuda.synchronize()
            persistent = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()

            def update(value: torch.Tensor) -> torch.Tensor:
                f0, onset = decode(value)
                audio = synth(f0, onset)
                scalar = objective.values(audio)[loss].sum()
                gradient, = torch.autograd.grad(scalar, value)
                return (value - 1e-4 * gradient).detach().requires_grad_(True)

            for _ in range(warmup):
                raw = update(raw)
            durations = []
            for _ in range(repeats):
                torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(measured):
                    raw = update(raw)
                torch.cuda.synchronize()
                durations.append((time.perf_counter() - started) / measured)

            peak = torch.cuda.max_memory_allocated()
            median_seconds = statistics.median(durations)
            ordered = sorted(durations)
            q1 = ordered[max(0, math.floor((len(ordered) - 1) * 0.25))]
            q3 = ordered[min(len(ordered) - 1, math.ceil((len(ordered) - 1) * 0.75))]
            row = {
                "loss": loss,
                "label": label,
                "cardinality": cardinality,
                "batch": batch,
                "median_ms_per_update": 1000.0 * median_seconds,
                "iqr_ms_per_update": 1000.0 * (q3 - q1),
                "peak_total_mib": peak / 2**20,
                "peak_incremental_mib": (peak - persistent) / 2**20,
                "median_recovery_updates": medians[(label, cardinality)],
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            del objective, raw
            gc.collect()
            torch.cuda.empty_cache()

    payload = {
        "schema": "loss-efficiency-benchmark-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "dtype": "float64",
        "sample_rate_hz": 16000,
        "duration_seconds": 2,
        "batch": batch,
        "warmup_updates": warmup,
        "measured_updates_per_repeat": measured,
        "repeats": repeats,
        "cardinalities": cardinalities,
        "timed_path": "render + bound objective + backward to pitch/onset logits",
        "target_precomputation_timed": False,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measured", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--cardinalities", type=int, nargs="+", choices=CARDINALITIES, default=[1]
    )
    args = parser.parse_args()
    benchmark(
        args.output,
        batch=args.batch,
        warmup=args.warmup,
        measured=args.measured,
        repeats=args.repeats,
        cardinalities=tuple(dict.fromkeys(args.cardinalities)),
    )
