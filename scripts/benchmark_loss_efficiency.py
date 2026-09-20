#!/usr/bin/env python3
"""Benchmark the end-to-end cost of the four representative paper losses."""

from __future__ import annotations

import gc
import json
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
    "single_stft": "SS",
    "smooth_mss": "SmoMSS",
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


def benchmark(
    output: Path,
    *,
    batch: int,
    warmup: int,
    measured: int,
    repeats: int,
    target_count: int,
    cardinalities: tuple[int, ...],
    losses: tuple[str, ...],
) -> None:
    device = torch.device("cuda")
    require_df2_backend(device)
    torch.set_num_threads(1)
    synth = PhraseSynth().to(device)
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    if target_count % batch:
        raise ValueError("target count must be divisible by batch size")

    for cardinality in cardinalities:
        initial = initial_candidate(cardinality, device=device)
        for loss in losses:
            label = LOSSES[loss]
            loss_rows = []
            for begin in range(0, target_count, batch):
                phrases = [
                    load_target(cardinality, index + 1, device=device)[1]
                    for index in range(begin, begin + batch)
                ]
                with torch.no_grad():
                    targets = synth(
                        torch.stack([phrase.f0_hz for phrase in phrases]),
                        torch.stack([phrase.onset_seconds for phrase in phrases]),
                    )
                objective = PaperObjectives(targets, (loss,))
                raw = encode(
                    initial.f0_hz.expand(batch, -1),
                    initial.onset_seconds.expand(batch, -1),
                ).clone().requires_grad_(True)
                torch.cuda.synchronize()
                persistent = torch.cuda.memory_allocated()

                def update(
                    value: torch.Tensor,
                    bound: PaperObjectives = objective,
                    name: str = loss,
                ) -> torch.Tensor:
                    f0, onset = decode(value)
                    audio = synth(f0, onset)
                    scalar = bound.values(audio)[name].sum()
                    gradient, = torch.autograd.grad(scalar, value)
                    return (value - 1e-4 * gradient).detach().requires_grad_(True)

                for _ in range(warmup):
                    raw = update(raw)
                durations = []
                peaks = []
                for _ in range(repeats):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    for _ in range(measured):
                        raw = update(raw)
                    torch.cuda.synchronize()
                    durations.append((time.perf_counter() - started) / measured)
                    peaks.append(torch.cuda.max_memory_allocated())

                row = {
                    "loss": loss,
                    "label": label,
                    "cardinality": cardinality,
                    "cell": begin // batch,
                    "target_begin": begin,
                    "target_end": begin + batch,
                    "batch": batch,
                    "mean_ms_per_update": 1000.0 * statistics.mean(durations),
                    "mean_peak_total_mib": statistics.mean(peaks) / 2**20,
                    "mean_peak_incremental_mib": (
                        statistics.mean(peaks) - persistent
                    ) / 2**20,
                }
                rows.append(row)
                loss_rows.append(row)
                print(json.dumps({"cell": row}), flush=True)
                del objective, raw, targets, phrases, update
                gc.collect()
                torch.cuda.empty_cache()

            times = [float(row["mean_ms_per_update"]) for row in loss_rows]
            memories = [float(row["mean_peak_total_mib"]) for row in loss_rows]
            summary = {
                "loss": loss,
                "label": label,
                "cardinality": cardinality,
                "cells": len(loss_rows),
                "targets": target_count,
                "batch": batch,
                "mean_ms_per_update": statistics.mean(times),
                "sample_std_ms_per_update": statistics.stdev(times),
                "median_ms_per_update": statistics.median(times),
                "mean_peak_total_mib": statistics.mean(memories),
                "sample_std_peak_total_mib": statistics.stdev(memories),
                "median_peak_total_mib": statistics.median(memories),
            }
            summaries.append(summary)
            print(json.dumps({"summary": summary}), flush=True)

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
        "target_count": target_count,
        "cardinalities": cardinalities,
        "losses": losses,
        "timed_path": "render + bound objective + backward to pitch/onset logits",
        "target_precomputation_timed": False,
        "rows": rows,
        "summaries": summaries,
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
    parser.add_argument("--target-count", type=int, default=150)
    parser.add_argument(
        "--cardinalities", type=int, nargs="+", choices=CARDINALITIES, default=[1]
    )
    parser.add_argument("--losses", nargs="+", choices=LOSSES, default=list(LOSSES))
    args = parser.parse_args()
    benchmark(
        args.output,
        batch=args.batch,
        warmup=args.warmup,
        measured=args.measured,
        repeats=args.repeats,
        target_count=args.target_count,
        cardinalities=tuple(dict.fromkeys(args.cardinalities)),
        losses=tuple(dict.fromkeys(args.losses)),
    )
