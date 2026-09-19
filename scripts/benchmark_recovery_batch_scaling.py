"""Measure recovery update throughput as the paired target batch grows."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from icassp27_phrase.config import initial_candidate
from icassp27_phrase.runtime import require_df2_backend
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.targets import load_target
from run_nine_loss_recovery import LOSSES, PairedObjective, decode, encode, signature


def run(cardinality: int, batches: tuple[int, ...], output: Path) -> None:
    device = "cuda"
    require_df2_backend(device)
    synth = PhraseSynth().to(device)
    rows = []
    for batch in batches:
        phrases = [load_target(cardinality, index + 1, device=device)[1] for index in range(batch)]
        with torch.no_grad():
            targets = synth(
                torch.stack([phrase.f0_hz for phrase in phrases]),
                torch.stack([phrase.onset_seconds for phrase in phrases]),
            )
        for name in LOSSES:
            objective = PairedObjective(targets, name)
            initial = initial_candidate(cardinality, device=device)
            raw = (
                encode(
                    initial.f0_hz.expand(batch, -1),
                    initial.onset_seconds.expand(batch, -1),
                )
                .clone()
                .requires_grad_(True)
            )
            warmup, measured = 3, 12
            torch.cuda.reset_peak_memory_stats()
            for update in range(warmup + measured):
                if update == warmup:
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                f0, onset = decode(raw)
                value = objective(synth(f0, onset)).sum()
                (gradient,) = torch.autograd.grad(value, raw)
                raw = (raw - 1e-4 * gradient).detach().requires_grad_(True)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            row = {
                "loss": name,
                "cardinality": cardinality,
                "batch": batch,
                "seconds_per_update": seconds / measured,
                "projected_minutes_20000_updates": seconds / measured * 20_000 / 60,
                "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    sig, hashes = signature()
    payload = {
        "schema": "phrase-recovery-16k-batch-scaling-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "signature": sig,
        "source_hashes": hashes,
        "gpu": torch.cuda.get_device_name(),
        "cardinality": cardinality,
        "batches": list(batches),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cardinality", type=int, required=True)
    parser.add_argument("--batches", default="150")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.cardinality, tuple(map(int, args.batches.split(","))), args.output)
