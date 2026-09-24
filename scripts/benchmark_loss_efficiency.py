#!/usr/bin/env python3
"""Benchmark render–loss–backward at identical, fixed candidate parameters."""

from __future__ import annotations

import gc
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from icassp27_phrase.data import load_target
from icassp27_phrase.losses import PaperObjectives
from icassp27_phrase.optimization import decode, encode
from icassp27_phrase.paths import resolve_path
from icassp27_phrase.runtime import require_df2_backend
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.synth.config import CARDINALITIES, initial_candidate

LOSSES = {
    "single_stft": "SS",
    "mss": "MSS",
    "smooth_mss": "SmoMSS",
    "sot_published_composite": "SOT",
    "linear_jtfot": "TFW2",
    "cel": "CeL",
}


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
    if min(batch, measured, repeats, target_count) < 1 or warmup < 0:
        raise ValueError("Counts must be positive; warmup may be zero")
    device = torch.device("cuda")
    if torch.cuda.get_device_name() != "NVIDIA A100-PCIE-40GB":
        raise RuntimeError("Benchmark requires the same A100 40 GB model for all rows")
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
                initial_raw = encode(
                    initial.f0_hz.expand(batch, -1),
                    initial.onset_seconds.expand(batch, -1),
                ).clone()

                raw = initial_raw.clone().requires_grad_(True)

                def evaluate(raw=raw, objective=objective, loss=loss):
                    f0, onset = decode(raw)
                    values = objective.values(synth(f0, onset))[loss]
                    (gradient,) = torch.autograd.grad(values.sum(), raw)
                    return values.detach(), gradient.detach()

                # Each pass builds and frees a fresh graph; no parameter update
                # or gradient accumulation occurs, including during warm-up.
                for _ in range(warmup):
                    evaluate()
                torch.cuda.synchronize()
                persistent = torch.cuda.memory_allocated()
                durations, peaks = [], []
                for _ in range(repeats):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    for _ in range(measured):
                        evaluate()
                    torch.cuda.synchronize()
                    durations.append((time.perf_counter() - started) / measured)
                    peaks.append(torch.cuda.max_memory_allocated())

                # Validation is outside timing and peak-memory capture.
                if not torch.equal(raw.detach(), initial_raw) or raw.grad is not None:
                    raise RuntimeError("Fixed candidate parameters or gradient state changed")
                values, gradient = evaluate()
                if not (torch.isfinite(values).all() and torch.isfinite(gradient).all()):
                    raise RuntimeError("Non-finite loss or gradient")
                f0, onset = decode(initial_raw)
                diagnostics = {
                    "parameters_unchanged": True,
                    "f0_hz": f0.tolist(),
                    "onset_seconds": onset.tolist(),
                    "loss_values": values.tolist(),
                    "gradient_finite": True,
                }
                del values, gradient, f0, onset

                row = {
                    "loss": loss,
                    "diagnostics": diagnostics,
                    "label": label,
                    "cardinality": cardinality,
                    "cell": begin // batch,
                    "target_begin": begin,
                    "target_end": begin + batch,
                    "batch": batch,
                    "mean_ms_per_pass": 1000.0 * statistics.mean(durations),
                    "mean_peak_total_mib": statistics.mean(peaks) / 2**20,
                    "mean_peak_incremental_mib": (statistics.mean(peaks) - persistent) / 2**20,
                }
                rows.append(row)
                loss_rows.append(row)
                print(json.dumps({"cell": row}), flush=True)
                del objective, initial_raw, raw, targets, phrases, evaluate
                gc.collect()
                torch.cuda.empty_cache()

            times = [float(row["mean_ms_per_pass"]) for row in loss_rows]
            memories = [float(row["mean_peak_total_mib"]) for row in loss_rows]
            summary = {
                "loss": loss,
                "label": label,
                "cardinality": cardinality,
                "cells": len(loss_rows),
                "targets": target_count,
                "batch": batch,
                "mean_ms_per_pass": statistics.mean(times),
                "sample_std_ms_per_pass": statistics.stdev(times),
                "median_ms_per_pass": statistics.median(times),
                "mean_peak_total_mib": statistics.mean(memories),
                "sample_std_peak_total_mib": statistics.stdev(memories),
                "median_peak_total_mib": statistics.median(memories),
            }
            summaries.append(summary)
            print(json.dumps({"summary": summary}), flush=True)

    payload = {
        "schema": "loss-efficiency-fixed-v3",
        "created_utc": datetime.now(UTC).isoformat(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "dtype": "float64",
        "sample_rate_hz": 16000,
        "duration_seconds": 2,
        "batch": batch,
        "warmup_passes": warmup,
        "measured_passes_per_repeat": measured,
        "repeats": repeats,
        "target_count": target_count,
        "cardinalities": cardinalities,
        "losses": losses,
        "timed_path": (
            "render + bound objective + backward to pitch/onset logits (fixed candidates)"
        ),
        "target_precomputation_timed": False,
        "optimizer": None,
        "candidate_parameters_fixed": True,
        "gradient_accumulation": False,
        "rows": rows,
        "summaries": summaries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=resolve_path, required=True)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measured", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--target-count", type=int, default=150)
    parser.add_argument("--cardinalities", type=int, nargs="+", choices=CARDINALITIES, default=[1])
    parser.add_argument(
        "--losses",
        nargs="+",
        choices=LOSSES,
        default=["single_stft", "mss", "sot_published_composite", "linear_jtfot", "cel"],
    )
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
