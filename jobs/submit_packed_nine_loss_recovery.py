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
RESULTS = ROOT / "results/phrase-recovery-16k"
SAFETY_FACTOR = 1.05
POOL_LIMITS = {"gpushort": 58 * 60, "andrena": 12 * 60 * 60}


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
    timing_signatures = set()
    benchmark_gpus = set()
    for cardinality in (1, 2, 4, 6, 8):
        payload = json.loads(
            (RESULTS / f"batch-scaling-c{cardinality}.json").read_text()
        )
        if payload["schema"] != "phrase-recovery-16k-batch-scaling-v1":
            raise ValueError(f"cardinality-{cardinality} timing schema changed")
        timing_signatures.add(payload["signature"])
        benchmark_gpus.add(payload["gpu"])
        timing_rows.extend(payload["rows"])
    if timing_signatures != {scientific_signature}:
        raise ValueError("timing benchmarks do not match the current scientific inputs")
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
        if not math.isfinite(estimate) or estimate >= POOL_LIMITS["andrena"]:
            raise RuntimeError(f"task {task} estimate {estimate:.1f}s exceeds Andrena limit")
        tasks.append({"task": task, "estimated_seconds": estimate})
    pending = [
        row for row in tasks if not is_complete(row["task"], scientific_signature, execution_plan)
    ]
    probes = [probe(*pool) for pool in POOLS]
    available = [row for row in probes if row["available"]]
    if not available:
        raise RuntimeError(f"no eligible GPU pool: {probes}")
    eligible = available
    qos = limits()
    assignments = {row["partition"]: [] for row in eligible}
    loads = {row["partition"]: 0.0 for row in eligible}
    for item in sorted(pending, key=lambda row: (-row["estimated_seconds"], row["task"])):
        task_pools = [
            row
            for row in eligible
            if item["estimated_seconds"] < POOL_LIMITS[row["partition"]]
        ]
        if not task_pools:
            raise RuntimeError(f"task {item['task']} does not fit either queue limit")
        lane = min(
            task_pools,
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
        "timing_benchmark_signatures": sorted(timing_signatures),
        "timing_reuse": (
            "Per-update objective and renderer code are unchanged; checkpoint retention and "
            "scheduler changes do not affect the measured update cost."
        ),
        "benchmark_gpus": sorted(benchmark_gpus),
        "safety_factor": SAFETY_FACTOR,
        "pool_time_limits_seconds": POOL_LIMITS,
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
        arguments = [
            "sbatch",
            "--parsable",
            f"--partition={partition}",
            f"--account={account}",
            f"--time={POOL_LIMITS[partition] // 60}",
            f"--array={array}%{qos[partition]}",
            (
                "--export=ALL,"
                f"SOURCE_COMMIT={source_commit},EXECUTION_PLAN_SHA256={execution_plan}"
            ),
        ]
        if partition == "gpushort":
            arguments.append("--constraint=ampere")
        arguments.append("jobs/packed_nine_loss_recovery.sh")
        result = command(arguments)
        jobs[partition] = result.stdout.strip().split(";", 1)[0]
    plan["jobs"] = jobs
    (RESULTS / "submission.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
