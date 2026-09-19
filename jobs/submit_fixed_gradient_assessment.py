#!/usr/bin/env python3
"""Submit the 16-kHz gradient assessment across both GPU pools."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from submit_nine_loss_recovery import POOLS, compress, limits, probe

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/gradient-assessment-16k"
CARDINALITIES = (1, 2, 4, 6, 8)
TARGETS_PER_CARDINALITY = 32


def command(arguments: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(arguments, cwd=ROOT, text=True, capture_output=True, check=check)


def task_coordinates(task: int) -> tuple[int, int]:
    return CARDINALITIES[task // TARGETS_PER_CARDINALITY], task % TARGETS_PER_CARDINALITY


def task_complete(task: int, signature: str) -> bool:
    events, index = task_coordinates(task)
    path = RESULTS / f"execution-c{events}-t{index:03d}.json"
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return payload.get("complete") is True and payload.get("signature") == signature


def main() -> None:
    if command(["git", "diff", "--quiet", "HEAD", "--"], check=False).returncode:
        raise RuntimeError("tracked source changes must be committed before submission")
    source_commit = command(["git", "rev-parse", "HEAD"]).stdout.strip()
    scripts_path = str(ROOT / "scripts")
    if scripts_path not in os.sys.path:
        os.sys.path.insert(0, scripts_path)
    from fixed_gradient_assessment import signature

    scientific_signature = signature()[0]
    probes = [probe(*pool) for pool in POOLS]
    available = [row for row in probes if row["available"]]
    if len(available) != len(POOLS):
        raise RuntimeError(f"both requested GPU pools must be available: {probes}")
    qos = limits()
    earliest = min(available, key=lambda row: row["estimated_start_epoch"])
    qualification = command(
        [
            "sbatch",
            "--parsable",
            f"--partition={earliest['partition']}",
            f"--account={earliest['account']}",
            (
                "--export=ALL,MODE=qualify,"
                f"SOURCE_COMMIT={source_commit},SCIENTIFIC_SIGNATURE={scientific_signature},"
                "BATCH_SIZE=16"
            ),
            "jobs/fixed_gradient_assessment.sh",
        ]
    ).stdout.strip().split(";", 1)[0]

    pending = [
        task
        for task in range(len(CARDINALITIES) * TARGETS_PER_CARDINALITY)
        if not task_complete(task, scientific_signature)
    ]
    assignments = {row["partition"]: [] for row in available}
    loads = {row["partition"]: 0.0 for row in available}
    conditions = {1: 3, 2: 3, 4: 3, 6: 1, 8: 1}
    weights = {
        task: conditions[task_coordinates(task)[0]]
        * (1.0 + 0.15 * task_coordinates(task)[0])
        for task in pending
    }
    for task in sorted(pending, key=lambda item: (-weights[item], item)):
        lane = min(
            available,
            key=lambda row: (
                loads[row["partition"]] / qos[row["partition"]],
                row["partition"],
            ),
        )["partition"]
        assignments[lane].append(task)
        loads[lane] += weights[task]

    jobs = {}
    for row in available:
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
                f"--dependency=afterok:{qualification}",
                f"--array={array}%{qos[partition]}",
                (
                    "--export=ALL,MODE=compute,"
                    f"SOURCE_COMMIT={source_commit},SCIENTIFIC_SIGNATURE={scientific_signature},"
                    "BATCH_SIZE=16"
                ),
                "jobs/fixed_gradient_assessment.sh",
            ]
        )
        jobs[partition] = result.stdout.strip().split(";", 1)[0]
    plan = {
        "schema": "gradient-assessment-16k-submission-v1",
        "source_commit": source_commit,
        "scientific_signature": scientific_signature,
        "probes": probes,
        "qos_gpu_limits": qos,
        "maximum_concurrent_gpus": sum(qos[row["partition"]] for row in available),
        "qualification_job": qualification,
        "pending_tasks": len(pending),
        "assignments": assignments,
        "relative_loads": loads,
        "jobs": jobs,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "submission.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
