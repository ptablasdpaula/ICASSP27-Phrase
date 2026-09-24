"""Dispatch a Slurm array element through the same local experiment commands."""

import argparse
import os
import subprocess
import sys

from icassp27_phrase.paths import OUTPUT, REPO, scientific_signature

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("experiment", choices=("gradients", "recovery", "efficiency", "slices", "qualify"))
p.add_argument("--task", type=int, default=0)
a = p.parse_args()
expected = os.environ.get("PHRASE_SOURCE_SIGNATURE")
if expected is not None and scientific_signature()[0] != expected:
    raise RuntimeError("Source changed after submission; regenerate the job submission")
if a.experiment == "gradients":
    args = ["run_gradients.py", "compute", "--task", str(a.task)]
elif a.experiment == "recovery":
    args = ["run_recovery.py", "--task", str(a.task)]
elif a.experiment == "qualify":
    args = ["run_gradients.py", "qualify"]
elif a.experiment == "slices":
    args = ["render_loss_sweeps.py"]
else:
    losses = ("single_stft", "mss", "sot_published_composite", "linear_jtfot", "cel")
    if not 0 <= a.task < len(losses):
        p.error("Invalid efficiency task")
    args = [
        "benchmark_loss_efficiency.py",
        "--losses",
        losses[a.task],
        "--output",
        str(OUTPUT / "efficiency" / f"{losses[a.task]}.json"),
    ]
subprocess.run([sys.executable, str(REPO / "scripts" / args[0]), *args[1:]], check=True)
