#!/usr/bin/env python3
"""Move only still-pending benchmark tasks after the first array completes."""
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run(*args):
    return subprocess.check_output(args, text=True, cwd=ROOT).strip()

records = []
for task in (2, 3, 4):
    job = f"28141107_{task}"
    queued = dict(line.split("|", 1) for line in run(
        "squeue", "-h", "-r", "-u", "acw794", "-o", "%i|%T"
    ).splitlines() if "|" in line)
    state = queued.get(job, "not queued")
    if state != "PENDING":
        records.append({"job": job, "action": "leave unchanged", "state": state})
        continue
    # The state filter prevents cancellation if scheduling wins this race.
    subprocess.run(["scancel", "--state=PENDING", job], check=True)
    cancelled = False
    for _ in range(12):
        states = run("sacct", "-X", "-n", "-P", "-j", job, "--format=State").splitlines()
        if states and all(s.startswith("CANCELLED") for s in states):
            cancelled = True
            break
        time.sleep(2)
    if not cancelled:
        records.append({"job": job, "action": "no replacement; cancellation not confirmed"})
        continue
    replacement = run("sbatch", "--parsable", f"--array={task}", "--partition=gpushort", "--account=pilot", "jobs/loss_efficiency_benchmark.sh")
    records.append({"job": job, "action": "moved to gpushort", "replacement": replacement})
    print(json.dumps(records[-1]), flush=True)

(ROOT / "results/loss-efficiency/queue-fallback.json").write_text(json.dumps(records, indent=2) + "\n")
print(json.dumps(records), flush=True)
