#!/usr/bin/env python3
"""Submit the fixed gradient campaign to the earliest eligible GPU pool."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from submit_nine_loss_recovery import POOLS, probe

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/gradient-assessment-fixed-lhs"


def command(arguments: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(arguments, cwd=ROOT, text=True, capture_output=True, check=check)


def main() -> None:
    if command(["git", "diff", "--quiet", "HEAD", "--"], check=False).returncode:
        raise RuntimeError("tracked source changes must be committed before submission")
    source_commit = command(["git", "rev-parse", "HEAD"]).stdout.strip()
    probes = [probe(*pool) for pool in POOLS]
    available = [row for row in probes if row["available"]]
    if not available:
        raise RuntimeError(f"no eligible GPU pool: {probes}")
    selected = min(available, key=lambda row: (row["estimated_start_epoch"], row["partition"]))
    result = command(
        [
            "sbatch",
            "--parsable",
            f"--partition={selected['partition']}",
            f"--account={selected['account']}",
            f"--export=ALL,SOURCE_COMMIT={source_commit}",
            "jobs/fixed_gradient_assessment.sh",
        ]
    )
    plan = {
        "schema": "fixed-gradient-assessment-submission-v1",
        "source_commit": source_commit,
        "probes": probes,
        "selected_pool": selected,
        "job_id": result.stdout.strip().split(";", 1)[0],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "submission.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
