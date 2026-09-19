#!/usr/bin/env python3
"""Submit the 16-kHz recovery qualification and A100 timing benchmark."""

from __future__ import annotations

import json
import os

from submit_nine_loss_recovery import POOLS, ROOT, command, limits, probe

RESULTS = ROOT / "results/phrase-recovery-16k"


def main() -> None:
    if command(["git", "diff", "--quiet", "HEAD", "--"], check=False).returncode:
        raise RuntimeError("tracked source changes must be committed before submission")
    source_commit = command(["git", "rev-parse", "HEAD"]).stdout.strip()
    scripts_path = str(ROOT / "scripts")
    if scripts_path not in os.sys.path:
        os.sys.path.insert(0, scripts_path)
    from run_nine_loss_recovery import signature

    scientific_signature = signature()[0]
    probes = [probe(*pool) for pool in POOLS]
    andrena = next(
        (row for row in probes if row["partition"] == "andrena" and row["available"]),
        None,
    )
    if andrena is None:
        raise RuntimeError(f"Andrena is unavailable: {probes}")
    qos = limits()
    qualification = command(
        [
            "sbatch",
            "--parsable",
            (
                "--export=ALL,"
                f"SOURCE_COMMIT={source_commit},SCIENTIFIC_SIGNATURE={scientific_signature}"
            ),
            "jobs/qualify_packed_nine_loss_recovery.sh",
        ]
    ).stdout.strip().split(";", 1)[0]
    benchmark = command(
        [
            "sbatch",
            "--parsable",
            f"--array=0-4%{min(5, qos['andrena'])}",
            (
                "--export=ALL,"
                f"SOURCE_COMMIT={source_commit},SCIENTIFIC_SIGNATURE={scientific_signature}"
            ),
            "jobs/benchmark_nine_loss_recovery.sh",
        ]
    ).stdout.strip().split(";", 1)[0]
    payload = {
        "schema": "phrase-recovery-16k-preflight-submission-v1",
        "source_commit": source_commit,
        "scientific_signature": scientific_signature,
        "probes": probes,
        "qos_gpu_limits": qos,
        "qualification_job": qualification,
        "benchmark_job": benchmark,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "preflight-submission.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
