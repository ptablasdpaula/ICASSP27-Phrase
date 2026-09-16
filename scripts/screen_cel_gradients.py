"""Run, qualify and summarise the independent CeL gradient screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from icassp27_phrase.cel_screen import (
    SCHEMA,
    candidates,
    compute_target,
    design,
    elementary,
    score,
    source_hash,
    subset_matrix,
)
from icassp27_phrase.losses import CEL_DIRECTIONS, CEL_NAMES, CumulativeEnergyDistance
from icassp27_phrase.runtime import configure_reproducibility, require_df2_backend
from icassp27_phrase.synth import PhraseSynth


def qualify(output: Path):
    results = {}
    arrays = {}
    for device in ("cpu", "cuda"):
        require_df2_backend(device)
        synth = PhraseSynth().to(device)
        for n in (1, 8):
            target = next(
                t for t in design() if len(t.coordinates) == n and t.profile == "independent"
            )
            positions, _ = candidates(target)
            x = torch.tensor(target.coordinates, dtype=torch.float64, device=device)
            with torch.no_grad():
                audio = synth(80 * 4 ** x[None, :, 0], 0.2 + 1.6 * x[None, :, 1])[0]
            start = time.perf_counter()
            arrays[device, n] = elementary(
                synth, CumulativeEnergyDistance(audio), positions[:4], device
            )
            results[f"{device}_{n}_seconds"] = time.perf_counter() - start
    for n in (1, 8):
        for i, key in enumerate(("loss", "gradient")):
            cpu, gpu = arrays["cpu", n][i], arrays["cuda", n][i]
            np.testing.assert_allclose(cpu, gpu, rtol=1e-6, atol=1e-8)
            results[f"{n}_{key}_max_abs_difference"] = float(np.max(np.abs(cpu - gpu)))
    results.update(
        source_hash=source_hash(),
        torch_version=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        passed=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results), flush=True)


def report(root: Path, output: Path, qualification: Path | None = None):
    output.mkdir(parents=True, exist_ok=True)
    groups = {}
    quality = {"candidates": 0, "ambiguous_assignments": 0, "nonfinite_variant_candidates": 0}
    metadata = []
    for target in design():
        path = root / f"{target.name}.npz"
        with np.load(path) as archive:
            data = dict(archive)
        if str(data["source_hash"]) != source_hash():
            raise ValueError(f"stale shard {path}")
        # An unrelated nonfinite direction must not contaminate a subset via 0*NaN.
        grad = np.stack(
            [data["elementary_gradients"][:, row > 0].mean(axis=1) for row in subset_matrix()],
            axis=1,
        )
        delta = data["target"][data["assignments"]] - data["candidates"]
        scores = score(grad, delta)
        tied = data["tied"]
        quality["candidates"] += len(tied)
        quality["ambiguous_assignments"] += int(tied.sum())
        quality["nonfinite_variant_candidates"] += int(scores["nonfinite"].sum())
        metadata.append(
            {
                "target": target.name,
                "profile": target.profile,
                "candidates": len(tied),
                "ambiguous": int(tied.sum()),
                "device": str(data["device"]),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        for kind in np.unique(data["kinds"]):
            selected = (data["kinds"] == kind) & ~tied
            for metric, values in scores.items():
                if metric == "nonfinite":
                    continue
                for profile in {
                    target.profile,
                    "independent" if target.profile == "independent" else "structured",
                }:
                    key = (len(target.coordinates), profile, str(kind), metric)
                    count = np.isfinite(values[selected]).sum(0)
                    means = np.divide(
                        np.nansum(values[selected], axis=0),
                        count,
                        out=np.full(30, np.nan),
                        where=count > 0,
                    )
                    groups.setdefault(key, []).append(means)
    rows = []
    for (n, profile, kind, metric), values in sorted(groups.items()):
        values = np.stack(values)
        for v, name in enumerate(CEL_NAMES):
            finite = values[:, v][np.isfinite(values[:, v])]
            rows.append(
                dict(
                    events=n,
                    profile=profile,
                    perturbation=kind,
                    metric=metric,
                    variant=name,
                    phrases=len(finite),
                    mean=float(finite.mean()) if len(finite) else None,
                    std=float(finite.std(ddof=1)) if len(finite) > 1 else None,
                    median=float(np.median(finite)) if len(finite) else None,
                )
            )
    with (output / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        "schema": SCHEMA,
        "source_hash": source_hash(),
        "numpy_version": np.__version__,
        "report_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "devices": sorted({item["device"] for item in metadata}),
        "seed": 2028,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "directions": CEL_DIRECTIONS,
        "variants": CEL_NAMES,
        "subset_matrix": subset_matrix().tolist(),
        "quality": quality,
        "shards": metadata,
        "aggregation": (
            "candidate scores averaged within target/type, then "
            "mean/sample std/median across targets"
        ),
        "tie_tolerance": 1e-12,
        "correct_coordinate_tolerance": 1e-12,
    }
    if qualification is not None:
        check = json.loads(qualification.read_text())
        if not check["passed"] or check["source_hash"] != source_hash():
            raise ValueError("qualification failed or is stale")
        provenance["cpu_gpu_qualification"] = check
    else:
        provenance["cpu_gpu_qualification"] = "not supplied; see per-shard execution devices"
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    with zipfile.ZipFile(output / "raw-results.zip", "w", compression=zipfile.ZIP_STORED) as z:
        for target in design():
            p = root / f"{target.name}.npz"
            z.write(p, arcname=p.name)
    heatmaps(rows, output)
    write_report(rows, quality, output)
    print(json.dumps(quality), flush=True)


def heatmaps(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ("pitch_directed", "onset_directed", "joint_directed")
    labels = []
    arrows = ("↗", "↘", "↖", "↙")
    for name in CEL_NAMES:
        mask = int(name.split("_")[1])
        labels.append(
            " ".join(a for i, a in enumerate(arrows) if mask & (1 << i))
            + (" + LW" if name.endswith("_lw") else "")
        )
    for profile, kind in [
        ("independent", "simultaneous"),
        ("structured", "isolated_joint"),
        ("structured", "isolated_timing"),
        ("structured", "isolated_pitch"),
        ("independent", "initialisation"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(10, 10), sharey=True, layout="constrained")
        for ax, metric in zip(axes, metrics, strict=True):
            matrix = np.full((30, 5), np.nan)
            for row in rows:
                if (
                    row["profile"] == profile
                    and row["perturbation"] == kind
                    and row["metric"] == metric
                ):
                    matrix[
                        CEL_NAMES.index(row["variant"]), (1, 2, 4, 6, 8).index(row["events"])
                    ] = np.nan if row["mean"] is None else row["mean"] * 100
            im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=100, cmap="viridis")
            ax.set_xticks(range(5), (1, 2, 4, 6, 8))
            ax.set_yticks(range(30), labels, fontsize=8)
            ax.set_xlabel("Events")
            ax.set_title(metric.replace("_directed", "").capitalize())
        fig.colorbar(im, ax=axes, label="Target-directed events (%)", shrink=0.6)
        fig.suptitle(f"{profile.capitalize()} targets — {kind.replace('_', ' ')}")
        fig.savefig(output / f"{profile}-{kind}.pdf")
        fig.savefig(output / f"{profile}-{kind}.png", dpi=130)
        plt.close(fig)


def write_report(rows, quality, output):
    lookup = {
        (r["events"], r["profile"], r["perturbation"], r["metric"], r["variant"]): r for r in rows
    }
    text = [
        "# CeL gradient screening at 2.048× onset padding",
        "",
        "All 30 direction subsets/Log-Weighing configurations were tested without optimisation. "
        "The square-root feature and all other synthesis/loss settings are fixed.",
        "",
        f"{quality['candidates']} candidate phrases; "
        f"{quality['ambiguous_assignments']} tied Hungarian assignments excluded; "
        f"{quality['nonfinite_variant_candidates']} nonfinite variant/candidate gradients.",
        "",
        "Direction bits: 1 = right/up, 2 = right/down, 4 = left/up, 8 = left/down. "
        "Add bits to identify a subset; `_lw` adds Log-Weighing. `cel_15` is the "
        "existing four-direction variant.",
        "",
        "## Independent targets, simultaneous perturbations",
        "",
        "Entries are mean target-directed event percentages, separately averaged "
        "within each phrase "
        "then across 30 independent targets at each event count. Each cell is "
        "pitch / onset / joint. "
        "The complete CSV also includes sample standard deviations, medians, cosine alignment, "
        "whole-phrase alignment, zero gradients and drift of already-correct coordinates.",
        "",
        "| Directions | 1 event | 2 events | 4 events | 6 events | 8 events |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in CEL_NAMES:
        cells = []
        for n in (1, 2, 4, 6, 8):
            vals = [
                lookup[n, "independent", "simultaneous", metric, name]["mean"]
                for metric in ("pitch_directed", "onset_directed", "joint_directed")
            ]
            cells.append(" / ".join(f"{100 * v:.1f}" for v in vals))
        mask = int(name.split("_")[1])
        label = " ".join(a for i, a in enumerate(("↗", "↘", "↖", "↙")) if mask & (1 << i))
        if name.endswith("_lw"):
            label += " + LW"
        text.append("| " + label + " | " + " | ".join(cells) + " |")
    text.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "These are local derivatives in normalised physical coordinates, not "
            "bounded logits or Adam updates. "
            "The implemented onset-state resets remain detached: gradients are "
            "conditional on the current discrete "
            "reset/order configuration. Positive alignment does not establish "
            "convergence or finite-step improvement.",
            "",
            "Single-event slices hold other events correct; the drift statistic "
            "measures gradients that move "
            "already-correct coordinates. Whole-phrase alignment can be positive "
            "even when some events or axes move away.",
            "",
            "Pilot targets use seed 2028 and are independent of the frozen "
            "150-target evaluation registry. "
            "No optimisation runs or final variant selection have been made. "
            "Existing phrase-recovery results "
            "remain from the earlier padding setting.",
            "",
            "Raw elementary losses/gradients, target/candidate coordinates, "
            "assignments and backend provenance "
            "are in `raw-results.zip`; `provenance.json` records hashes and subset coefficients. "
            "Reconstruct each variant by averaging its elementary terms. Blank "
            "summary cells denote no eligible "
            "coordinates or insufficient phrases for sample standard deviation.",
        ]
    )
    (output / "README.md").write_text("\n".join(text) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("compute", "qualify", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--index", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--qualification", type=Path)
    args = parser.parse_args()
    configure_reproducibility()
    if args.command == "qualify":
        qualify(args.output)
    elif args.command == "report":
        report(args.input, args.output, args.qualification)
    else:
        require_df2_backend(args.device)
        if args.device.startswith("cuda"):
            if args.qualification is None:
                raise ValueError("CUDA campaign requires a CPU/GPU qualification report")
            qualification = json.loads(args.qualification.read_text())
            if not qualification["passed"] or qualification["source_hash"] != source_hash():
                raise ValueError("qualification failed or is stale")
        indices = (
            [args.index]
            if args.index is not None
            else range(args.start, len(design()), args.stride)
        )
        for index in indices:
            start = time.perf_counter()
            path = compute_target(index, args.output, args.device, args.chunk_size)
            print(f"{index}: {path.name} ({time.perf_counter() - start:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
