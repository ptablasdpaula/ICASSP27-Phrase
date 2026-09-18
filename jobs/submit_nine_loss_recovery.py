#!/usr/bin/env python3
"""Runtime-balance and submit the qualified nine-loss recovery campaign."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/nine-loss-recovery"
POOLS = (
    ("gpushort", "pilot"),
    ("andrena", "pilot_andrena"),
)
START_WINDOW_SECONDS = 15 * 60
SAFETY_FACTOR = 1.2
HARD_SECONDS = 55 * 60


def command(arguments, *, check=True):
    return subprocess.run(arguments, cwd=ROOT, text=True, capture_output=True, check=check)


def probe(partition, account):
    result = command(
        [
            "sbatch",
            "--test-only",
            f"--partition={partition}",
            f"--account={account}",
            "--gres=gpu:1",
            "--time=00:59:00",
            "--cpus-per-task=2",
            "--mem=32G",
            "--wrap=/bin/true",
        ],
        check=False,
    )
    match = re.search(r"to start at (\S+).*in partition (\S+)", result.stdout)
    if result.returncode or match is None:
        return {
            "partition": partition,
            "account": account,
            "available": False,
            "diagnostic": result.stdout.strip(),
        }
    start = datetime.fromisoformat(match.group(1))
    if start.tzinfo is None:
        start = start.astimezone()
    return {
        "partition": partition,
        "account": account,
        "available": True,
        "estimated_start": start.isoformat(),
        "estimated_start_epoch": start.timestamp(),
        "diagnostic": result.stdout.strip(),
    }


def limits():
    result = command(
        [
            "sacctmgr",
            "-nP",
            "show",
            "qos",
            "gpushort,andrena",
            "format=Name,MaxTRESPU",
        ]
    )
    found = {}
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 2:
            continue
        match = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", fields[1])
        if match:
            found[fields[0]] = int(match.group(1))
    if not all(found.get(name, 0) > 0 for name, _ in POOLS):
        raise RuntimeError("could not determine live GPU QOS limits")
    return found


def compress(indices):
    ordered = sorted(indices)
    pieces = []
    if not ordered:
        return ""
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        pieces.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    pieces.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(pieces)


def expected_path(shard):
    from scripts.run_nine_loss_recovery import (
        ROOT as OUTPUT_ROOT,
    )
    from scripts.run_nine_loss_recovery import (
        shard_coordinates,
    )

    loss, cardinality, begin, count = shard_coordinates(shard)
    end = begin + count - 1
    return OUTPUT_ROOT / "raw" / loss / f"C{cardinality:02d}-T{begin:04d}-{end:04d}.json.gz"


def main():
    if command(["git", "diff", "--quiet", "HEAD", "--"], check=False).returncode:
        raise RuntimeError("tracked source changes must be committed before submission")
    source_commit = command(["git", "rev-parse", "HEAD"]).stdout.strip()
    benchmarks = [
        json.loads((RESULTS / name).read_text())
        for name in ("benchmark.json", "benchmark-batch5.json")
    ]
    qualification = json.loads((RESULTS / "qualification-cuda.json").read_text())
    sys_path = str(ROOT)
    if sys_path not in os.sys.path:
        os.sys.path.insert(0, sys_path)
    from scripts.run_nine_loss_recovery import (
        SHARD_SPECS,
        signature,
    )

    current_signature = signature()[0]
    if (
        any(row["signature"] != current_signature for row in benchmarks)
        or qualification["signature"] != current_signature
    ):
        raise ValueError("benchmark or qualification belongs to different scientific inputs")
    timings = {
        (row["loss"], int(row["cardinality"]), int(row["batch"])): float(
            row["projected_seconds_20000_updates"]
        )
        * SAFETY_FACTOR
        for benchmark in benchmarks
        for row in benchmark["rows"]
    }
    shards = []
    for shard, (loss, cardinality, _begin, count) in enumerate(SHARD_SPECS):
        estimate = timings[(loss, cardinality, count)]
        if not math.isfinite(estimate) or estimate >= HARD_SECONDS:
            raise RuntimeError(
                f"shard {shard} worst-case estimate {estimate:.1f}s exceeds sub-hour safety"
            )
        shards.append({"shard": shard, "estimated_seconds": estimate})
    pending = [row for row in shards if not expected_path(row["shard"]).exists()]
    probes = [probe(*pool) for pool in POOLS]
    available = [row for row in probes if row["available"]]
    if not available:
        raise RuntimeError(f"no eligible GPU pool: {probes}")
    earliest = min(row["estimated_start_epoch"] for row in available)
    eligible = [
        row for row in available if row["estimated_start_epoch"] <= earliest + START_WINDOW_SECONDS
    ]
    qos = limits()
    assignments = {row["partition"]: [] for row in eligible}
    loads = {row["partition"]: 0.0 for row in eligible}
    # Longest-processing-time placement, normalized by each pool's live GPU
    # allowance, adapts the upper-directory recovery sharding routine here.
    for item in sorted(pending, key=lambda row: (-row["estimated_seconds"], row["shard"])):
        lane = min(
            eligible,
            key=lambda row: (
                loads[row["partition"]] / qos[row["partition"]],
                row["partition"],
            ),
        )["partition"]
        assignments[lane].append(item["shard"])
        loads[lane] += item["estimated_seconds"]
    plan = {
        "schema": "nine-loss-recovery-shard-plan-v1",
        "source_commit": source_commit,
        "signature": current_signature,
        "benchmark_gpus": sorted({benchmark["gpu"] for benchmark in benchmarks}),
        "safety_factor": SAFETY_FACTOR,
        "worst_case_updates_per_fit": 20_000,
        "probes": probes,
        "qos_gpu_limits": qos,
        "eligible_pools": [row["partition"] for row in eligible],
        "pending_shards": len(pending),
        "assignments": assignments,
        "estimated_gpu_seconds_by_pool": loads,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "shard-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    jobs = {}
    for row in eligible:
        partition, account = row["partition"], row["account"]
        array = compress(assignments[partition])
        if not array:
            continue
        concurrency = qos[partition]
        result = command(
            [
                "sbatch",
                "--parsable",
                f"--partition={partition}",
                f"--account={account}",
                f"--array={array}%{concurrency}",
                f"--export=ALL,SOURCE_COMMIT={source_commit}",
                "jobs/nine_loss_recovery.sh",
            ]
        )
        jobs[partition] = result.stdout.strip().split(";", 1)[0]
    plan["jobs"] = jobs
    (RESULTS / "submission.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
