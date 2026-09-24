#!/usr/bin/env python3
"""Benchmark 100 actual Adam recovery updates for representative paper losses."""

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
from run_nine_loss_recovery import SCHEDULE, encode, decode

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
    if not 0 < measured <= 100:
        raise ValueError("This fixed-horizon benchmark supports 1–100 updates")
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

                def start_run():
                    value = initial_raw.clone().requires_grad_(True)
                    optimizer = torch.optim.Adam(
                        [value], lr=SCHEDULE.initial_lr, betas=SCHEDULE.betas,
                        eps=SCHEDULE.epsilon, weight_decay=0, foreach=False,
                    )
                    return value, optimizer

                def update(value, optimizer, state):
                    optimizer.zero_grad(set_to_none=True)
                    f0, onset = decode(value)
                    values = objective.values(synth(f0, onset))[loss]
                    detached = values.detach()
                    if not state:
                        state["initial"] = detached.clone()
                        state["best"] = detached.clone()
                        state["best_raw"] = value.detach().clone()
                    else:
                        better = detached < state["best"]
                        state["best"] = torch.minimum(state["best"], detached)
                        state["best_raw"] = torch.where(
                            better[:, None, None], value.detach(), state["best_raw"]
                        )
                    (values / state["initial"]).sum().backward()
                    optimizer.step()

                # Warm the kernels on disposable optimiser/parameter state.
                # Every measured run starts again at the registered initialisation.
                raw, optimizer = start_run()
                state = {}
                for _ in range(warmup):
                    update(raw, optimizer, state)
                del raw, optimizer, state
                torch.cuda.synchronize()
                persistent = torch.cuda.memory_allocated()
                durations, peaks, diagnostics = [], [], []
                for _ in range(repeats):
                    raw, optimizer = start_run()
                    state = {}
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    for _ in range(measured):
                        update(raw, optimizer, state)
                    torch.cuda.synchronize()
                    durations.append((time.perf_counter() - started) / measured)
                    peaks.append(torch.cuda.max_memory_allocated())
                    # Final evaluation is outside the measured update loop.
                    with torch.no_grad():
                        f0, onset = decode(raw)
                        final = objective.values(synth(f0, onset))[loss]
                    diagnostics.append({
                        "initial_loss": state["initial"].tolist(),
                        "final_loss": final.tolist(),
                        "best_loss": torch.minimum(state["best"], final).tolist(),
                        "final_f0_hz": f0.tolist(),
                        "final_onset_seconds": onset.tolist(),
                        "adam_steps": int(optimizer.state[raw]["step"].item()),
                    })
                    if not torch.isfinite(final).all():
                        raise RuntimeError("Non-finite loss after Adam updates")
                    del raw, optimizer, state

                row = {
                    "loss": loss,
                    "diagnostics": diagnostics,
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
                del objective, initial_raw, targets, phrases, update, start_run
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
        "schema": "loss-efficiency-adam-v2",
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
        "timed_path": "render + bound objective + initial-loss-normalised backward + Adam + best-iterate tracking",
        "target_precomputation_timed": False,
        "optimizer": {"name": "Adam", "lr": SCHEDULE.initial_lr, "betas": SCHEDULE.betas, "eps": SCHEDULE.epsilon, "weight_decay": 0},
        "warmup_state_discarded": True,
        "scheduler": "No reduction or early stop can occur within 100 updates (patience 200/1000)",
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
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measured", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--target-count", type=int, default=150)
    parser.add_argument(
        "--cardinalities", type=int, nargs="+", choices=CARDINALITIES, default=[1]
    )
    parser.add_argument("--losses", nargs="+", choices=LOSSES, default=["single_stft", "mss", "sot_published_composite", "linear_jtfot", "cel"])
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
