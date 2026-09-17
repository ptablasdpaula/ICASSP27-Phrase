"""Add joint six/eight-event columns without modifying the frozen core signature."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import assess_gradients as engine
import numpy as np
import torch
from icassp27_phrase.gradient_assessment import NAMES, SEED, candidates, seed_for, signature
from report_gradient_alignment import alignment

EXTENDED_COLUMNS = (
    (1, "joint"),
    (2, "joint"),
    (4, "joint"),
    (6, "joint"),
    (8, "joint"),
    (1, "pitch"),
    (2, "pitch"),
    (4, "pitch"),
    (1, "time"),
    (2, "time"),
    (4, "time"),
)


def extra_targets():
    registry = [("C01-T000", np.array([[1.0, 1.0]]))]
    for n in (6, 8):
        for i in range(32):
            name = f"C{n:02d}-T{i:03d}"
            rng = np.random.default_rng(seed_for("target", name))
            pitch = rng.uniform(0, 2, n)
            while True:
                onset = rng.uniform(0.2, 1.8, n)
                order = np.argsort(onset)
                if np.diff(onset[order]).min() >= 0.05:
                    break
            registry.append((name, np.stack((pitch[order], onset[order]), axis=-1)))
    return registry


def extended_candidates(name, target, count=256, repeat=0):
    design = candidates(name, target, count, repeat)
    if len(target) == 1:
        pitch, onset = design["joint"].copy(), design["joint"].copy()
        pitch[..., 1] = target[:, 1]
        onset[..., 0] = target[:, 0]
        design.update(pitch=pitch, time=onset)
    return design


def campaign(root, batch):
    started = time.perf_counter()
    registry = extra_targets()
    # The existing engine resolves its target registry dynamically. Only extend
    # this isolated process; preserve the core source and archived signatures.
    engine.targets = extra_targets
    engine.candidates = extended_candidates
    engine.setup("cuda")
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if (root / "design.json").exists():
        previous = json.loads((root / "design.json").read_text())
        if previous["extension_sha256"] != source_hash:
            raise ValueError("Extension source changed; use a fresh result root")
    engine.save_json(
        root / "design.json",
        dict(
            signature=signature()[0],
            extension_sha256=source_hash,
            seed=SEED,
            targets={name: t.tolist() for name, t in registry},
            losses=NAMES,
            columns=[[6, "joint"], [8, "joint"], [1, "pitch"], [1, "time"]],
            batch=batch,
            device="cuda",
            coordinates=["log2(f/80) octaves", "onset seconds"],
            matching="unrestricted Hungarian: squared octaves + squared seconds",
            sampling="256, then 512/1024 if original componentwise repeat sensitivity exceeds 2pp",
        ),
    )
    engine.save_json(
        root / "qualification.json",
        dict(
            signature=signature()[0],
            device="cuda",
            gpu=torch.cuda.get_device_name(),
            cpu_gpu_comparison="omitted at user request",
            checks="Every shard must contain finite losses and gradients",
        ),
    )
    active, counts, history = [(1, "pitch"), (1, "time"), (6, "joint"), (8, "joint")], {}, []
    for count in (256, 512, 1024):
        for index, (name, target) in enumerate(registry):
            conditions = [c for n, c in active if n == len(target)]
            if not conditions:
                continue
            for repeat in (0, 1) if int(name.split("T")[1]) < 8 else (0,):
                engine.compute_task((str(root), index, count, repeat, conditions, batch, "cuda"))
        following = []
        for n, condition in active:
            check = engine.sensitivity(root, (n, condition), count)
            pairs = [[], []]
            for name, target in registry:
                if len(target) != n or int(name.split("T")[1]) >= 8:
                    continue
                for repeat in (0, 1):
                    data = engine.checked_data(
                        engine.shard_path(root, name, condition, count, repeat)
                    )
                    pairs[repeat].append(np.nanmean(alignment(data)["phrase-cosine"], axis=0))
            means = np.mean(pairs, axis=1)
            check["phrase_cosine_repeat_absolute_difference"] = np.abs(means[0] - means[1]).tolist()
            history.append(check)
            print("SENSITIVITY", json.dumps(check), flush=True)
            if check["passed"] or count == 1024:
                counts[f"{n}-{condition}"] = count
            else:
                following.append((n, condition))
        engine.save_json(
            root / "sampling.json",
            dict(
                signature=signature()[0],
                final_counts=counts,
                checks=history,
                complete=not following,
                selection_metric="both_directed",
            ),
        )
        active = following
        if not active:
            break
    engine.save_json(
        root / "execution.json",
        dict(
            seconds=time.perf_counter() - started,
            gpu=torch.cuda.get_device_name(),
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            batch=batch,
            extension_sha256=source_hash,
            determinism="CUDA reflection-padding backward is not guaranteed bitwise deterministic",
        ),
    )
    print("EXTENSION COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/gradient-assessment-extended"))
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    campaign(args.root, args.batch)
    output = Path("docs/gradient-assessment/alignment-extended")
    subprocess.run(
        [
            sys.executable,
            "scripts/report_gradient_alignment.py",
            "--extension",
            str(args.root),
            "--output",
            str(output),
        ],
        check=True,
    )
    raw_output = output / "extension-data"
    subprocess.run(
        [
            sys.executable,
            "scripts/package_gradient_assessment.py",
            "--root",
            str(args.root),
            "--output",
            str(raw_output),
            "--job-id",
            os.environ.get("SLURM_JOB_ID", "manual"),
        ],
        check=True,
    )
    for name in ("design.json", "sampling.json", "qualification.json"):
        shutil.copy2(args.root / name, raw_output / name)
    shutil.copy2(args.root / "execution.json", raw_output / "campaign-execution.json")
    print("EXPORTED", output / "phrase-cosine.png", flush=True)
