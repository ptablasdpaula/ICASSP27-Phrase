"""Measure batched update cost for runtime-balanced production sharding."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from icassp27_phrase.config import CARDINALITIES, initial_candidate
from icassp27_phrase.runtime import require_df2_backend
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.targets import load_target
from run_nine_loss_recovery import LOSSES, PairedObjective, decode, encode, signature


def benchmark(warmup: int, measured: int, batch: int, output: Path) -> None:
    device = "cuda"
    require_df2_backend(device)
    synth = PhraseSynth().to(device)
    rows = []
    for name in LOSSES:
        for cardinality in CARDINALITIES:
            phrases = [
                load_target(cardinality, index + 1, device=device)[1] for index in range(batch)
            ]
            with torch.no_grad():
                targets = synth(
                    torch.stack([phrase.f0_hz for phrase in phrases]),
                    torch.stack([phrase.onset_seconds for phrase in phrases]),
                )
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
            torch.cuda.reset_peak_memory_stats()
            elapsed = None
            for update in range(warmup + measured):
                if update == warmup:
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                f0, onset = decode(raw)
                value = objective(synth(f0, onset)).sum()
                (gradient,) = torch.autograd.grad(value, raw)
                raw = (raw - 1e-4 * gradient).detach().requires_grad_(True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            row = {
                "loss": name,
                "cardinality": cardinality,
                "batch": batch,
                "measured_updates": measured,
                "seconds": elapsed,
                "seconds_per_update": elapsed / measured,
                "projected_seconds_1000_updates": elapsed / measured * 1000,
                "projected_seconds_20000_updates": elapsed / measured * 20000,
                "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    sig, hashes = signature()
    payload = {
        "schema": "phrase-recovery-16k-benchmark-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "signature": sig,
        "source_hashes": hashes,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "warmup_updates": warmup,
        "measured_updates": measured,
        "batch": batch,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--measured", type=int, default=12)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument(
        "--output", type=Path, default=Path("results/phrase-recovery-16k/benchmark.json")
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    benchmark(args.warmup, args.measured, args.batch, args.output)
