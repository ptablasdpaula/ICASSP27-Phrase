"""High-throughput execution plan for the qualified nine-loss recovery."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import run_nine_loss_recovery as recovery
import torch
from icassp27_phrase.config import CARDINALITIES
from icassp27_phrase.runtime import require_df2_backend

ROOT = Path("results/phrase-recovery-16k")


def targets_per_shard(loss: str, cardinality: int) -> int:
    if loss not in recovery.LOSSES or cardinality not in CARDINALITIES:
        raise ValueError("unsupported loss or cardinality")
    # Two A100-safe vectorised shards cover each 150-target cell.
    return recovery.TARGETS_PER_CELL // 2


def build_specs():
    specs = []
    for loss in recovery.LOSSES:
        for cardinality in CARDINALITIES:
            count = targets_per_shard(loss, cardinality)
            if recovery.TARGETS_PER_CELL % count:
                raise ValueError("packed shard size must divide 150 targets")
            specs.extend(
                (loss, cardinality, begin, count)
                for begin in range(0, recovery.TARGETS_PER_CELL, count)
            )
    return tuple(specs)


SHARD_SPECS = build_specs()
TOTAL_SHARDS = len(SHARD_SPECS)


def plan_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def expected_path(task: int):
    loss, cardinality, begin, count = SHARD_SPECS[task]
    return ROOT / "raw" / loss / f"C{cardinality:02d}-T{begin:04d}-{begin + count - 1:04d}.json.gz"


def run(task: int, device: str) -> None:
    recovery.ROOT = ROOT
    recovery.SHARD_SPECS = SHARD_SPECS
    recovery.TOTAL_SHARDS = TOTAL_SHARDS
    recovery.run(task, device)
    path = expected_path(task)
    with gzip.open(path, "rt") as stream:
        payload = json.load(stream)
    digest = plan_hash()
    recorded = payload.get("execution_plan_sha256")
    if recorded is not None and recorded != digest:
        raise ValueError(f"stale execution plan in {path}")
    if recorded is None:
        payload["execution_plan"] = "packed-nine-loss-recovery-v1"
        payload["execution_plan_sha256"] = digest
        payload["packed_task"] = task
        temporary = Path(str(path) + ".tmp")
        with gzip.open(temporary, "wt") as stream:
            json.dump(payload, stream, separators=(",", ":"), allow_nan=False)
        temporary.replace(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.task < TOTAL_SHARDS:
        raise ValueError(f"task must be in [0,{TOTAL_SHARDS})")
    torch.set_num_threads(1)
    require_df2_backend(args.device)
    run(args.task, args.device)
