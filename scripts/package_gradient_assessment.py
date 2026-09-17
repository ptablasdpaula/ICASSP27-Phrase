"""Archive completed gradient shards and execution metadata without rerendering."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from assess_gradients import checked_data, save_json
from icassp27_phrase.gradient_assessment import signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/gradient-assessment-gpu"))
    parser.add_argument("--output", type=Path, default=Path("docs/gradient-assessment"))
    parser.add_argument("--job-id", default="27248959")
    args = parser.parse_args()
    sampling = json.loads((args.root / "sampling.json").read_text())
    assert sampling["complete"] and sampling["signature"] == signature()[0]
    groups = defaultdict(list)
    total, seconds, peak_rss = 0, 0.0, 0
    raw_manifest = {}
    for path in sorted((args.root / "raw").glob("n*/r*/*.npz")):
        data = checked_data(path)
        name = str(data["target_id"])
        cardinality, index = name.split("-T")
        key = f"{path.parent.parent.name}-{cardinality}-group{int(index) // 8}"
        groups[key].append(path)
        total += len(data["candidates"])
        seconds += float(data["wall_seconds"])
        peak_rss = max(peak_rss, int(data["peak_rss_kib"]))
        raw_manifest[str(path.relative_to(args.root))] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "candidates": len(data["candidates"]),
            "device": str(data["device"]),
            "ties": int(data["ties"].sum()),
            "finite": bool(np.isfinite(data["gradients"]).all()),
        }
    archives = {}
    directory = args.output / "raw"
    directory.mkdir(parents=True, exist_ok=True)
    for name, paths in sorted(groups.items()):
        archive = directory / f"{name}.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as output:
            for path in paths:
                output.write(path, arcname=str(path.relative_to(args.root)))
        archives[str(archive.relative_to(args.output))] = {
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "bytes": archive.stat().st_size,
            "shards": len(paths),
        }
    save_json(
        args.output / "raw-manifest.json",
        {
            "signature": signature()[0],
            "archives": archives,
            "shards": raw_manifest,
            "extract_into": str(args.root),
        },
    )
    benchmarks = {}
    for root in (Path("results/gradient-assessment"), args.root):
        for path in sorted(root.glob("benchmark-b*.json")):
            benchmarks[f"{root.name}/{path.name}"] = json.loads(path.read_text())
    execution = dict(
        job_id=args.job_id,
        signature=signature()[0],
        backend="CUDA float64",
        gpu=json.loads((args.root / "qualification.json").read_text())["gpu"],
        all_saved_comparisons=total,
        all_saved_loss_gradient_evaluations=total * 11,
        summed_shard_seconds=seconds,
        peak_host_rss_kib=peak_rss,
        benchmarks=benchmarks,
        cpu_gpu_comparison="omitted at user request",
        determinism="CUDA reflection-padding backward is not guaranteed bitwise deterministic",
        gradient_coordinates=["log2(f/80) octaves", "onset seconds"],
        source_hashes={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path("scripts/assess_gradients.py"),
                Path(__file__),
                Path("jobs/gradient_assessment_gpu.sh"),
                Path("scripts/extend_gradient_cardinality.py"),
                Path("jobs/gradient_cardinality_gpu.sh"),
            )
        },
    )
    save_json(args.output / "execution.json", execution)
    print(f"Archived {len(raw_manifest)} shards, {total} comparisons, {len(archives)} archives")


if __name__ == "__main__":
    main()
