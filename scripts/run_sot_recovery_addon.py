#!/usr/bin/env python3
"""Run the matched SOT recovery control alongside the frozen main campaign."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

import run_nine_loss_recovery as recovery
import torch
from icassp27_phrase.config import CARDINALITIES
from icassp27_phrase.runtime import require_df2_backend
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.targets import load_target

ROOT = Path("results/phrase-recovery-16k-sot")
LOSS = "sot_published_composite"
LABEL = "SOT"
TARGETS_PER_SHARD = 75
SHARD_SPECS = tuple(
    (LOSS, cardinality, begin, TARGETS_PER_SHARD)
    for cardinality in CARDINALITIES
    for begin in range(0, recovery.TARGETS_PER_CELL, TARGETS_PER_SHARD)
)
TOTAL_SHARDS = len(SHARD_SPECS)


def plan_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def expected_path(task: int) -> Path:
    loss, cardinality, begin, count = SHARD_SPECS[task]
    return ROOT / "raw" / loss / f"C{cardinality:02d}-T{begin:04d}-{begin + count - 1:04d}.json.gz"


def configure_recovery() -> None:
    recovery.ROOT = ROOT
    recovery.LOSSES = (LOSS,)
    recovery.LABELS = (LABEL,)
    recovery.SHARD_SPECS = SHARD_SPECS
    recovery.TOTAL_SHARDS = TOTAL_SHARDS


def qualify(device: str) -> None:
    require_df2_backend(device)
    synth = PhraseSynth().to(device)
    metadata, phrases = zip(
        *(load_target(2, index, device=device) for index in (1, 2)), strict=True
    )
    del metadata
    with torch.no_grad():
        target = synth(
            torch.stack([phrase.f0_hz for phrase in phrases]),
            torch.stack([phrase.onset_seconds for phrase in phrases]),
        )
    f0 = torch.stack([phrase.f0_hz for phrase in phrases]) * 1.01
    onset = torch.stack([phrase.onset_seconds for phrase in phrases]) + 0.005
    raw = recovery.encode(f0.clamp(80.0, 320.0), onset.clamp(0.2, 1.8)).requires_grad_(True)
    candidate_f0, candidate_onset = recovery.decode(raw)
    candidate = synth(candidate_f0, candidate_onset)
    objective = recovery.PairedObjective(target, LOSS)(candidate)
    (gradient,) = torch.autograd.grad(objective.sum(), raw)
    if not bool(torch.isfinite(objective).all() and torch.isfinite(gradient).all()):
        raise FloatingPointError("SOT qualification produced non-finite values")
    signature, hashes = recovery.signature()
    payload = {
        "passed": True,
        "scientific_signature": signature,
        "source_hashes": hashes,
        "execution_plan_sha256": plan_hash(),
        "total_shards": TOTAL_SHARDS,
        "total_fits": len(CARDINALITIES) * recovery.TARGETS_PER_CELL,
        "device": device,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "objective": objective.detach().cpu().tolist(),
        "gradient_max_abs": float(gradient.detach().abs().max()),
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "qualification.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def run(task: int, device: str) -> None:
    configure_recovery()
    recovery.run(task, device)
    path = expected_path(task)
    with gzip.open(path, "rt") as stream:
        payload = json.load(stream)
    payload["schema"] = "phrase-recovery-16k-sot-addon-v1"
    payload["execution_plan"] = "sot-recovery-addon-v1"
    payload["execution_plan_sha256"] = plan_hash()
    payload["addon_task"] = task
    payload["addon_source_sha256"] = plan_hash()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + path.name, suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with gzip.open(temporary, "wt") as stream:
            json.dump(payload, stream, separators=(",", ":"), allow_nan=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("qualify", "compute"))
    parser.add_argument("--task", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "qualify":
        qualify(args.device)
        return
    if args.task is None or not 0 <= args.task < TOTAL_SHARDS:
        raise ValueError(f"task must be in [0,{TOTAL_SHARDS})")
    run(args.task, args.device)


if __name__ == "__main__":
    main()
