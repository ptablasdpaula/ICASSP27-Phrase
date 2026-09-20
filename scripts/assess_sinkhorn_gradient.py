#!/usr/bin/env python3
"""Evaluate full time--frequency Sinkhorn in the paper's cosine diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import resource
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from assess_limitation_columns import (
    CONTROL_NAMES,
    all_control_candidates,
    all_control_targets,
    cosine,
    persistent_targets,
    render_all_controls,
)
from fixed_gradient_assessment import (
    CANDIDATES_PER_TARGET,
    CARDINALITIES,
    COLUMNS,
    TARGETS_PER_CARDINALITY,
    candidates,
    targets,
)
from icassp27_phrase.config import WaveguideConfig
from icassp27_phrase.metrics import hungarian_assignment, phrase_gradient_cosine
from icassp27_phrase.runtime import configure_reproducibility, require_df2_backend
from icassp27_phrase.synth import PhraseSynth
from sinkhorn_tfw import SinkhornTFWObjective

ROOT = Path("results/gradient-assessment-sinkhorn")
OUTPUT = Path("docs/gradient-assessment/sinkhorn")
SCHEMA = "gradient-assessment-sinkhorn-v2"
STANDARD = "standard"
PERSISTENT = "persistent_target"
ALL_CONTROLS = "all_controls"
CONDITIONS = (STANDARD, PERSISTENT, ALL_CONTROLS)


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def setup(device: str) -> None:
    torch.set_num_threads(1)
    configure_reproducibility()
    require_df2_backend(device)


def signature() -> tuple[str, dict[str, str]]:
    paths = [
        Path(__file__),
        Path(__file__).with_name("sinkhorn_tfw.py"),
        Path(__file__).with_name("fixed_gradient_assessment.py"),
        Path(__file__).with_name("assess_limitation_columns.py"),
        *(Path(__file__).parents[1] / "src" / name for name in (
            "losses.py", "metrics.py", "synth.py", "exciter.py", "waveguide.py",
            "_fractional.py", "config.py", "runtime.py",
        )),
    ]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), hashes


def standard_path(root: Path, events: int, condition: str, target_index: int) -> Path:
    return root / "raw" / f"{condition}-{events}" / f"target-{target_index:03d}.npz"


def limitation_path(root: Path, condition: str, target_index: int) -> Path:
    return root / "raw" / condition / f"target-{target_index:03d}.npz"


def _evaluate(
    objective: SinkhornTFWObjective,
    coordinates: torch.Tensor,
    audio: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    loss = objective(audio)
    (gradient,) = torch.autograd.grad(loss.sum(), coordinates)
    values = loss.detach().cpu().numpy()
    gradients = gradient.detach().cpu().numpy()
    if not np.isfinite(values).all() or not np.isfinite(gradients).all():
        raise FloatingPointError("non-finite Sinkhorn value or gradient")
    return values, gradients


def _save(path: Path, **values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary, schema=SCHEMA, signature=signature()[0], **values
    )
    temporary.replace(path)


def _batched_standard(
    objective: SinkhornTFWObjective,
    synth: PhraseSynth,
    positions: np.ndarray,
    batch: int,
) -> tuple[np.ndarray, ...]:
    blocks = ([], [])
    for begin in range(0, len(positions), batch):
        coordinate = torch.tensor(
            positions[begin : begin + batch],
            dtype=torch.float64,
            device=objective.target.device,
            requires_grad=True,
        )
        audio = synth(80.0 * 2.0 ** coordinate[..., 0], coordinate[..., 1])
        result = _evaluate(objective, coordinate, audio)
        for destination, value in zip(blocks, result, strict=True):
            destination.append(value)
    return tuple(np.concatenate(value) for value in blocks)


def compute_standard(
    root: Path, events: int, target_index: int, batch: int, device: str
) -> None:
    setup(device)
    selected = [(name, target) for name, target in targets() if len(target) == events]
    name, target = selected[target_index]
    synth = PhraseSynth().to(device)
    target_tensor = torch.tensor(target, dtype=torch.float64, device=device)
    with torch.no_grad():
        target_audio = synth(
            80.0 * 2.0 ** target_tensor[None, :, 0], target_tensor[None, :, 1]
        )[0]
        objective = SinkhornTFWObjective(target_audio)
    design = candidates(name, target)
    for _, condition in (column for column in COLUMNS if column[0] == events):
        path = standard_path(root, events, condition, target_index)
        if path.exists():
            saved = _load(path)
            if not np.array_equal(saved["target"], target) or not np.array_equal(
                saved["candidates"], design[condition]
            ):
                raise ValueError(f"stale Sinkhorn design: {path}")
            continue
        started = time.perf_counter()
        positions = design[condition]
        values, gradients = _batched_standard(
            objective, synth, positions, batch
        )
        assignments, ties = zip(
            *(hungarian_assignment(row, target) for row in positions), strict=True
        )
        _save(
            path,
            condition=condition,
            events=events,
            target_id=name,
            target=target,
            candidates=positions,
            losses=values,
            gradients=gradients,
            assignments=np.stack(assignments),
            ties=np.asarray(ties),
            objective_json=json.dumps(objective.provenance(), sort_keys=True),
            renderer_json=json.dumps(synth.provenance(), sort_keys=True),
            wall_seconds=time.perf_counter() - started,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        print(
            f"DONE {name} {condition}: n={len(values)}",
            flush=True,
        )


def compute_persistent(root: Path, target_index: int, batch: int, device: str) -> None:
    setup(device)
    name, target = persistent_targets()[target_index]
    positions = candidates(name, target)["joint"]
    target_tensor = torch.tensor(target, dtype=torch.float64, device=device)
    target_synth = PhraseSynth(
        waveguide_config=replace(WaveguideConfig(), state_policy="persistent")
    ).to(device)
    candidate_synth = PhraseSynth().to(device)
    with torch.no_grad():
        target_audio = target_synth(
            80.0 * 2.0 ** target_tensor[None, :, 0], target_tensor[None, :, 1]
        )[0]
        objective = SinkhornTFWObjective(target_audio)
    started = time.perf_counter()
    values, gradients = _batched_standard(
        objective, candidate_synth, positions, batch
    )
    assignments, ties = zip(
        *(hungarian_assignment(row, target) for row in positions), strict=True
    )
    _save(
        limitation_path(root, PERSISTENT, target_index),
        condition=PERSISTENT,
        target_id=name,
        target=target,
        candidates=positions,
        losses=values,
        gradients=gradients,
        assignments=np.stack(assignments),
        ties=np.asarray(ties),
        objective_json=json.dumps(objective.provenance(), sort_keys=True),
        wall_seconds=time.perf_counter() - started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    print(f"DONE {PERSISTENT} {target_index:03d}", flush=True)


def compute_all_controls(root: Path, target_index: int, batch: int, device: str) -> None:
    setup(device)
    name, target = all_control_targets()[target_index]
    positions = all_control_candidates(name)
    target_tensor = torch.tensor(target[None], dtype=torch.float64, device=device)
    with torch.no_grad():
        target_audio = render_all_controls(target_tensor)[0]
        objective = SinkhornTFWObjective(target_audio)
    blocks = ([], [])
    started = time.perf_counter()
    for begin in range(0, len(positions), batch):
        coordinate = torch.tensor(
            positions[begin : begin + batch],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        result = _evaluate(objective, coordinate, render_all_controls(coordinate))
        for destination, value in zip(blocks, result, strict=True):
            destination.append(value)
    values, gradients = (
        np.concatenate(value) for value in blocks
    )
    _save(
        limitation_path(root, ALL_CONTROLS, target_index),
        condition=ALL_CONTROLS,
        target_id=name,
        target=target,
        candidates=positions,
        losses=values,
        gradients=gradients,
        control_names=np.asarray(CONTROL_NAMES),
        objective_json=json.dumps(objective.provenance(), sort_keys=True),
        wall_seconds=time.perf_counter() - started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    print(f"DONE {ALL_CONTROLS} {target_index:03d}", flush=True)


def qualify(root: Path, device: str, batch: int) -> None:
    setup(device)
    name, target = next((name, row) for name, row in targets() if len(row) == 1)
    positions = candidates(name, target)["joint"][:batch]
    synth = PhraseSynth().to(device)
    target_tensor = torch.tensor(target, dtype=torch.float64, device=device)
    with torch.no_grad():
        target_audio = synth(
            80.0 * 2.0 ** target_tensor[None, :, 0], target_tensor[None, :, 1]
        )[0]
        objective = SinkhornTFWObjective(target_audio)
    started = time.perf_counter()
    values, gradients = _batched_standard(
        objective, synth, positions, batch
    )
    gradient_norms = np.linalg.norm(gradients.reshape(len(gradients), -1), axis=1)
    if not np.isfinite(gradient_norms).all() or bool((gradient_norms == 0.0).any()):
        raise RuntimeError("Sinkhorn qualification produced an invalid gradient")
    save_json(
        root / "qualification.json",
        {
            "passed": True,
            "signature": signature()[0],
            "source_hashes": signature()[1],
            "device": device,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
            "batch": batch,
            "seconds": time.perf_counter() - started,
            "seconds_per_candidate": (time.perf_counter() - started) / len(positions),
            "gradient_norms": gradient_norms.tolist(),
            "losses": values.tolist(),
            "objective": objective.provenance(),
        },
    )
    print(json.dumps(json.loads((root / "qualification.json").read_text()), indent=2))


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as saved:
        data = dict(saved)
    if str(data["schema"]) != SCHEMA or str(data["signature"]) != signature()[0]:
        raise ValueError(f"stale Sinkhorn shard: {path}")
    return data


def report(root: Path, output: Path) -> None:
    rows: list[dict[str, object]] = []
    artifacts: dict[str, str] = {}
    registry = {events: [] for events in CARDINALITIES}
    for name, target in targets():
        registry[len(target)].append((name, target))
    for events, condition in COLUMNS:
        blocks = []
        for index, (_, target) in enumerate(registry[events]):
            path = standard_path(root, events, condition, index)
            data = _load(path)
            blocks.append(
                phrase_gradient_cosine(
                    data["gradients"][:, None], data["candidates"], target,
                    data["assignments"], data["ties"],
                )[:, 0]
            )
            artifacts[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        values = np.concatenate(blocks)
        valid = values[np.isfinite(values)]
        rows.append({
            "condition": condition,
            "events": events,
            "mean": float(valid.mean()),
            "sample_sd": float(valid.std(ddof=1)),
            "positive_percent": float(100.0 * (valid > 0.0).mean()),
            "eligible": len(valid),
        })
    for condition in (PERSISTENT, ALL_CONTROLS):
        blocks = []
        target_rows = persistent_targets() if condition == PERSISTENT else all_control_targets()
        for index, (_, target) in enumerate(target_rows):
            path = limitation_path(root, condition, index)
            data = _load(path)
            if condition == PERSISTENT:
                values = phrase_gradient_cosine(
                    data["gradients"][:, None], data["candidates"], target,
                    data["assignments"], data["ties"],
                )[:, 0]
            else:
                values = cosine(
                    data["gradients"][:, None], data["candidates"], target
                )[:, 0]
            blocks.append(values)
            artifacts[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        combined = np.concatenate(blocks)
        valid = combined[np.isfinite(combined)]
        rows.append({
            "condition": condition,
            "events": 2 if condition == PERSISTENT else 1,
            "mean": float(valid.mean()),
            "sample_sd": float(valid.std(ddof=1)),
            "positive_percent": float(100.0 * (valid > 0.0).mean()),
            "eligible": len(valid),
        })
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cells = [f"{row['mean']:.3f} ± {row['sample_sd']:.3f}" for row in rows]
    headers = [
        *(f"{condition}: {events}" for events, condition in COLUMNS),
        "State: 2", "7-D: 1",
    ]
    markdown = [
        "# Full Sinkhorn time-frequency OT gradient cosine",
        "",
        "| Loss | " + " | ".join(headers) + " |",
        "|---|" + "---:|" * len(headers),
        "| Sinkhorn | " + " | ".join(cells) + " |",
        "",
        (
            "All cells use 32 targets × 256 candidates. Values are mean "
            "phrase-wide cosine ± sample SD."
        ),
    ]
    (output / "summary.md").write_text("\n".join(markdown) + "\n")
    save_json(output / "provenance.json", {
        "schema": SCHEMA,
        "signature": signature()[0],
        "source_hashes": signature()[1],
        "paper": (
            "Fabiani, Schlecht, and Elvander, Time-Frequency Audio Similarity "
            "Using Optimal Transport, Asilomar 2024"
        ),
        "doi": "10.1109/IEEECONF60004.2024.10943074",
        "pairs_per_column": TARGETS_PER_CARDINALITY * CANDIDATES_PER_TARGET,
        "total_pairs": len(rows) * TARGETS_PER_CARDINALITY * CANDIDATES_PER_TARGET,
        "artifacts": artifacts,
    })
    print(output / "summary.md", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("qualify", "compute", "report"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--kind", choices=CONDITIONS, default=STANDARD)
    parser.add_argument("--cardinality", type=int, choices=CARDINALITIES)
    parser.add_argument("--target-index", type=int)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    if args.command == "qualify":
        qualify(args.root, args.device, args.batch)
    elif args.command == "report":
        report(args.root, args.output)
    else:
        if args.target_index is None:
            parser.error("compute requires --target-index")
        if args.kind == STANDARD:
            if args.cardinality is None:
                parser.error("standard compute requires --cardinality")
            compute_standard(
                args.root, args.cardinality, args.target_index, args.batch, args.device
            )
        elif args.kind == PERSISTENT:
            compute_persistent(args.root, args.target_index, args.batch, args.device)
        else:
            compute_all_controls(args.root, args.target_index, args.batch, args.device)


if __name__ == "__main__":
    main()
