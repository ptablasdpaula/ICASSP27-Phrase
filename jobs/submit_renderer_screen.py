#!/usr/bin/env python3
"""Adaptive launcher for the eight-loss supplementary renderer screen."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRATCH_BASE = Path("/gpfs/scratch/acw794/ICASSP27-Phrase/renderer-screen-eight-loss-v1")
CANDIDATES = (
    ("gpushort", "volta"),
    ("gpushort", "ampere"),
    ("andrena", "ampere"),
)


def run(
    command: list[str], *, timeout: int = 30, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )


def source_commit() -> str:
    status = run(["git", "status", "--porcelain=v1", "--untracked-files=all"], timeout=20).stdout
    if status.strip():
        raise RuntimeError("scientific inputs must be committed before submission")
    return run(["git", "rev-parse", "HEAD"], timeout=10).stdout.strip()


def probe(candidate: tuple[str, str]) -> dict[str, Any]:
    partition, constraint = candidate
    result = run(
        [
            "sbatch",
            "--test-only",
            f"--partition={partition}",
            "--account=pilot_andrena",
            "--gres=gpu:1",
            f"--constraint={constraint}",
            "--time=00:59:00",
            "--cpus-per-gpu=12",
            "--mem=64G",
            "--wrap=/bin/true",
        ],
        timeout=20,
        check=False,
    )
    match = re.search(r"to start at (\S+).*in partition (\S+)", result.stdout)
    if result.returncode or match is None:
        return {
            "partition": partition,
            "constraint": constraint,
            "available": False,
            "diagnostic": result.stdout.strip(),
        }
    start = datetime.fromisoformat(match.group(1))
    if start.tzinfo is None:
        start = start.astimezone()
    return {
        "partition": partition,
        "constraint": constraint,
        "available": True,
        "estimated_start": start.isoformat(),
        "estimated_start_epoch": start.timestamp(),
        "scheduler_partition": match.group(2),
        "diagnostic": result.stdout.strip(),
    }


def live_qos_limits() -> dict[str, int]:
    result = run(
        [
            "sacctmgr",
            "-nP",
            "show",
            "qos",
            "gpushort,andrena",
            "format=Name,MaxTRESPU",
        ],
        timeout=20,
    )
    limits: dict[str, int] = {}
    for line in result.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) < 2 or fields[0] not in {"gpushort", "andrena"}:
            continue
        match = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", fields[1])
        if match:
            limits[fields[0]] = int(match.group(1))
    if set(limits) != {"gpushort", "andrena"}:
        raise RuntimeError("could not resolve live GPU QOS limits")
    return limits


def submit(
    name: str,
    script: str,
    options: list[str],
    *,
    commit: str,
    output_root: Path,
    job_class: str,
) -> dict[str, Any]:
    (ROOT / "logs").mkdir(exist_ok=True)
    exports = {
        "ICASSP27_REPOSITORY": str(ROOT),
        "ICASSP27_SOURCE_COMMIT": commit,
        "ICASSP27_RENDERER_SCREEN_ROOT": str(output_root),
        "ICASSP27_JOB_CLASS": job_class,
    }
    if any("," in key or "," in value for key, value in exports.items()):
        raise ValueError("Slurm export values may not contain commas")
    command = [
        "sbatch",
        "--parsable",
        f"--job-name={name}",
        *options,
        "--export=ALL," + ",".join(f"{key}={value}" for key, value in exports.items()),
        str(ROOT / script),
    ]
    result = run(command)
    job_id = result.stdout.strip().split(";", 1)[0]
    if re.fullmatch(r"\d+", job_id) is None:
        raise RuntimeError(f"could not parse Slurm job id: {result.stdout}")
    start = run(
        [
            "squeue",
            "--start",
            "--jobs",
            job_id,
            "--format=%i|%P|%T|%S|%R",
            "--noheader",
        ],
        timeout=15,
        check=False,
    ).stdout.strip()
    return {"job_id": job_id, "command": command, "start_report": start}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    commit = source_commit()
    output_root = SCRATCH_BASE / commit
    probes = list(ThreadPoolExecutor(max_workers=3).map(probe, CANDIDATES))
    available = sorted(
        (value for value in probes if value["available"]),
        key=lambda value: value["estimated_start_epoch"],
    )
    if not available:
        raise RuntimeError("no registered GPU partition accepted a probe")
    partitions = list(dict.fromkeys(value["partition"] for value in available))
    limits = live_qos_limits()
    concurrency = min(16, sum(limits[name] for name in partitions))
    gpu_options = [
        "--partition=" + ",".join(partitions),
        "--account=pilot_andrena",
        "--gres=gpu:1",
        "--constraint=[volta|ampere]",
        "--time=00:59:00",
        "--ntasks=1",
        "--cpus-per-gpu=12",
        "--mem=64G",
        "--requeue",
        f"--array=0-15%{concurrency}",
        "--output=" + str(ROOT / "logs/%x-%A_%a.out"),
    ]
    cpu_options = [
        "--partition=computeshort",
        "--account=pilot",
        "--time=00:30:00",
        "--cpus-per-task=2",
        "--mem=8G",
        "--output=" + str(ROOT / "logs/%x-%j.out"),
    ]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "source_commit": commit,
                    "output_root": str(output_root),
                    "probes": probes,
                    "qos_limits": limits,
                    "partitions": partitions,
                    "array_concurrency": concurrency,
                    "gpu_options": gpu_options,
                    "cpu_options": cpu_options,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    cells = submit(
        "ic27-renderer-screen",
        "jobs/renderer_screen_cell.sh",
        gpu_options,
        commit=commit,
        output_root=output_root,
        job_class="gpu",
    )
    assembled = submit(
        "ic27-renderer-screen-build",
        "jobs/renderer_screen_aggregate.sh",
        [*cpu_options, f"--dependency=afterok:{cells['job_id']}"],
        commit=commit,
        output_root=output_root,
        job_class="cpu",
    )
    print(
        json.dumps(
            {
                "source_commit": commit,
                "output_root": str(output_root),
                "probes": probes,
                "qos_limits": limits,
                "partitions": partitions,
                "array_concurrency": concurrency,
                "cells": cells,
                "assembly": assembled,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
