#!/usr/bin/env python3
"""Submit the measured high-throughput recovery plan to the live GPU pools."""

from __future__ import annotations

import gzip
import json
import math
import os
from pathlib import Path

from submit_nine_loss_recovery import POOLS, command, compress, limits, probe

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/nine-loss-recovery-packed"
START_WINDOW_SECONDS = 15 * 60
SAFETY_FACTOR = 1.05
HARD_SECONDS = 58 * 60


def is_complete(task, expected_signature, expected_plan):
    from scripts.run_packed_nine_loss_recovery import expected_path

    path = expected_path(task)
    if not path.exists():
        return False
    try:
        with gzip.open(path, "rt") as stream:
            payload = json.load(stream)
        return (
            payload["signature"] == expected_signature
            and payload["execution_plan_sha256"] == expected_plan
        )
    except (OSError, KeyError, ValueError):
        return False


def main():
    if command(["git", "diff", "--quiet", "HEAD", "--"], check=False).returncode:
        raise RuntimeError("tracked source changes must be committed before submission")
    source_commit = command(["git", "rev-parse", "HEAD"]).stdout.strip()
    sys_path = str(ROOT)
    if sys_path not in os.sys.path:
        os.sys.path.insert(0, sys_path)
    scripts_path = str(ROOT / "scripts")
    if scripts_path not in os.sys.path:
        os.sys.path.insert(0, scripts_path)
    from scripts.run_nine_loss_recovery import signature
    from scripts.run_packed_nine_loss_recovery import SHARD_SPECS, plan_hash

    scientific_signature = signature()[0]
    execution_plan = plan_hash()
    qualification = json.loads((RESULTS / "qualification-cuda.json").read_text())
    if (
        not qualification["passed"]
        or qualification["scientific_signature"] != scientific_signature
        or qualification["execution_plan_sha256"] != execution_plan
    ):
        raise ValueError("packed qualification does not match current inputs")
    timing_rows = []
    benchmark_gpus = set()
    for cardinality in (1, 2, 4, 6, 8):
        payload = json.loads(
            (ROOT / f"results/nine-loss-recovery/batch-scaling-c{cardinality}.json").read_text()
        )
        if payload["signature"] != scientific_signature:
            raise ValueError(f"cardinality-{cardinality} timing signature changed")
        benchmark_gpus.add(payload["gpu"])
        timing_rows.extend(payload["rows"])
    timings = {
        (row["loss"], int(row["cardinality"]), int(row["batch"])): float(
            row["projected_minutes_20000_updates"]
        )
        * 60
        * SAFETY_FACTOR
        for row in timing_rows
    }
    tasks = []
    for task, (loss, cardinality, _begin, count) in enumerate(SHARD_SPECS):
        estimate = timings[(loss, cardinality, count)]
        if not math.isfinite(estimate) or estimate >= HARD_SECONDS:
            raise RuntimeError(f"task {task} estimate {estimate:.1f}s exceeds packed limit")
        tasks.append({"task": task, "estimated_seconds": estimate})
    pending = [
        row for row in tasks if not is_complete(row["task"], scientific_signature, execution_plan)
    ]
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
    for item in sorted(pending, key=lambda row: (-row["estimated_seconds"], row["task"])):
        lane = min(
            eligible,
            key=lambda row: (
                loads[row["partition"]] / qos[row["partition"]],
                row["partition"],
            ),
        )["partition"]
        assignments[lane].append(item["task"])
        loads[lane] += item["estimated_seconds"]
    plan = {
        "schema": "packed-nine-loss-recovery-shard-plan-v1",
        "source_commit": source_commit,
        "scientific_signature": scientific_signature,
        "execution_plan_sha256": execution_plan,
        "benchmark_gpus": sorted(benchmark_gpus),
        "safety_factor": SAFETY_FACTOR,
        "hard_seconds": HARD_SECONDS,
        "task_count": len(tasks),
        "pending_tasks": len(pending),
        "probes": probes,
        "qos_gpu_limits": qos,
        "eligible_pools": [row["partition"] for row in eligible],
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
        result = command(
            [
                "sbatch",
                "--parsable",
                f"--partition={partition}",
                f"--account={account}",
                f"--array={array}%{qos[partition]}",
                f"--export=ALL,EXECUTION_PLAN_SHA256={execution_plan}",
                "jobs/packed_nine_loss_recovery.sh",
            ]
        )
        jobs[partition] = result.stdout.strip().split(";", 1)[0]
    plan["jobs"] = jobs
    (RESULTS / "submission.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
