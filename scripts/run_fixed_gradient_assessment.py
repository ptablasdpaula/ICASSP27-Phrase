#!/usr/bin/env python3
"""Run and report the paper's fixed-LHS thirteen-loss gradient assessment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import resource
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from fixed_gradient_assessment import (
    CANDIDATES_PER_TARGET,
    CARDINALITIES,
    COLUMNS,
    NAMES,
    SCHEMA,
    TARGETS_PER_CARDINALITY,
    FixedObjectives,
    candidates,
    design_metadata,
    signature,
    targets,
)
from icassp27_phrase.losses import CEL_DIRECTIONS
from icassp27_phrase.metrics import hungarian_assignment, phrase_gradient_cosine
from icassp27_phrase.runtime import configure_reproducibility, require_df2_backend
from icassp27_phrase.synth import PhraseSynth
from report_gradient_alignment import compact_cosine, render

ROOT = Path("results/gradient-assessment-16k")
OUTPUT = Path("docs/gradient-assessment/16k")
PAPER_FIGURE = Path("paper/figures/gradient_alignment.pdf")
LABELS = (
    r"$L_1$",
    r"$L_2$",
    "SS",
    "MSS",
    "SmoMSS",
    "SOT",
    "SOT-NC",
    r"$\mathrm{TF}\mathcal{W}_2$",
    r"log$\mathrm{TF}\mathcal{W}_2$",
    r"Ce$\mathcal{L}$",
    r"logCe$\mathcal{L}$",
    r"decCe$\mathcal{L}$",
    r"tlogCe$\mathcal{L}$",
)
TEXT_LABELS = (
    "L1",
    "L2",
    "SS",
    "MSS",
    "SmoMSS",
    "SOT",
    "SOT-NC",
    "TFW2",
    "logTFW2",
    "CeL",
    "logCeL",
    "decCeL",
    "tlogCeL",
)


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def setup(device: str) -> None:
    torch.set_num_threads(1)
    configure_reproducibility()
    require_df2_backend(device)


def registry() -> dict[int, list[tuple[str, np.ndarray]]]:
    grouped = {events: [] for events in CARDINALITIES}
    for name, target in targets():
        grouped[len(target)].append((name, target))
    if any(len(rows) != TARGETS_PER_CARDINALITY for rows in grouped.values()):
        raise ValueError("fixed target registry is incomplete")
    return grouped


def conditions(events: int) -> tuple[str, ...]:
    return tuple(condition for cardinality, condition in COLUMNS if cardinality == events)


def raw_path(root: Path, events: int, condition: str, name: str) -> Path:
    return root / "raw" / f"{condition}-{events}" / f"{name}.npz"


def checked_data(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as saved:
        data = dict(saved)
    count, events, dimensions = data["candidates"].shape
    if str(data["signature"]) != signature()[0]:
        raise ValueError(f"stale shard: {path}")
    if (
        str(data["schema"]) != SCHEMA
        or count != CANDIDATES_PER_TARGET
        or dimensions != 2
        or tuple(data["names"]) != NAMES
        or data["losses"].shape != (count, len(NAMES))
        or data["gradients"].shape != (count, len(NAMES), events, 2)
        or data["assignments"].shape != (count, events)
        or data["ties"].shape != (count,)
    ):
        raise ValueError(f"incomplete shard: {path}")
    if not np.isfinite(data["losses"]).all() or not np.isfinite(data["gradients"]).all():
        raise FloatingPointError(path)
    return data


def qualify(root: Path, device: str, batch: int) -> None:
    setup(device)
    assert CEL_DIRECTIONS[:2] == ("right_up", "right_down")
    target_registry = registry()
    for events, rows in target_registry.items():
        assert len(rows) == TARGETS_PER_CARDINALITY
        for name, target in rows:
            design = candidates(name, target)
            assert len(design["joint"]) == CANDIDATES_PER_TARGET
            if events > 1:
                assert np.diff(target[:, 1]).min() >= 0.05 - 1e-12
    name, target = target_registry[2][0]
    design = candidates(name, target)["joint"][:batch]
    synth = PhraseSynth().to(device)
    coordinates = torch.tensor(target, dtype=torch.float64, device=device)
    with torch.no_grad():
        target_audio = synth(80 * 2 ** coordinates[None, :, 0], coordinates[None, :, 1])[0]
    objective = FixedObjectives(target_audio)
    values, gradients = objective.evaluate(synth, design)
    assert values.shape == (len(design), len(NAMES))
    assert gradients.shape == (len(design), len(NAMES), 2, 2)
    assert np.isfinite(values).all() and np.isfinite(gradients).all()
    save_json(
        root / "qualification.json",
        {
            "passed": True,
            "signature": signature()[0],
            "source_hashes": signature()[1],
            "device": device,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
            "checks": [
                "32 deterministic LHS targets at every cardinality",
                "50-ms target onset separation",
                "256 deterministic LHS candidates per target",
                "forward directions are right_up and right_down",
                "finite values and gradients for all thirteen losses",
            ],
        },
    )
    save_json(root / "design.json", design_metadata())


def compute_cardinality(
    root: Path,
    events: int,
    batch: int,
    device: str,
    target_index: int | None = None,
) -> None:
    setup(device)
    qualification = json.loads((root / "qualification.json").read_text())
    if not qualification["passed"] or qualification["signature"] != signature()[0]:
        raise ValueError("fixed-LHS qualification is missing or stale")
    synth = PhraseSynth().to(device)
    started = time.perf_counter()
    completed = 0
    selected = registry()[events]
    if target_index is not None:
        if not 0 <= target_index < len(selected):
            raise ValueError("target_index is outside the fixed target registry")
        selected = [selected[target_index]]
    for name, target in selected:
        design = candidates(name, target)
        target_coordinates = torch.tensor(target, dtype=torch.float64, device=device)
        with torch.no_grad():
            target_audio = synth(
                80 * 2 ** target_coordinates[None, :, 0], target_coordinates[None, :, 1]
            )[0]
            objective = FixedObjectives(target_audio)
        for condition in conditions(events):
            output = raw_path(root, events, condition, name)
            positions = design[condition]
            if output.exists():
                data = checked_data(output)
                if not np.array_equal(data["target"], target) or not np.array_equal(
                    data["candidates"], positions
                ):
                    raise ValueError(f"stale design in {output}")
                continue
            task_started = time.perf_counter()
            values, gradients = [], []
            for begin in range(0, CANDIDATES_PER_TARGET, batch):
                value, gradient = objective.evaluate(synth, positions[begin : begin + batch])
                values.append(value)
                gradients.append(gradient)
            assignments, ties = zip(
                *(hungarian_assignment(row, target) for row in positions), strict=True
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary,
                schema=SCHEMA,
                signature=signature()[0],
                target=target,
                candidates=positions,
                losses=np.concatenate(values),
                gradients=np.concatenate(gradients),
                assignments=np.stack(assignments),
                ties=np.array(ties),
                names=np.array(NAMES),
                condition=condition,
                target_id=name,
                renderer_json=json.dumps(synth.provenance(), sort_keys=True),
                torch_version=torch.__version__,
                device=device,
                wall_seconds=time.perf_counter() - task_started,
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            )
            temporary.replace(output)
            completed += CANDIDATES_PER_TARGET
            print(
                f"DONE {name} {condition} n={CANDIDATES_PER_TARGET} "
                f"{time.perf_counter() - task_started:.1f}s",
                flush=True,
            )
    save_json(
        root
        / (
            f"execution-c{events}-t{target_index:03d}.json"
            if target_index is not None
            else f"execution-c{events}.json"
        ),
        {
            "complete": True,
            "signature": signature()[0],
            "events": events,
            "target_index": target_index,
            "new_pairs": completed,
            "seconds": time.perf_counter() - started,
            "batch": batch,
            "device": device,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
            "job_id": os.environ.get("SLURM_JOB_ID", "manual"),
        },
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(root: Path, output: Path, paper_figure: Path | None) -> None:
    target_registry = registry()
    means = np.zeros((len(NAMES), len(COLUMNS)))
    deviations = np.zeros_like(means)
    rows, quality, artifacts = [], [], {}
    total_pairs = 0
    for column, (events, condition) in enumerate(COLUMNS):
        blocks = []
        for name, target in target_registry[events]:
            path = raw_path(root, events, condition, name)
            data = checked_data(path)
            design = candidates(name, target)[condition]
            np.testing.assert_array_equal(data["target"], target)
            np.testing.assert_array_equal(data["candidates"], design)
            values = phrase_gradient_cosine(
                data["gradients"],
                data["candidates"],
                data["target"],
                data["assignments"],
                data["ties"],
            )
            blocks.append(values)
            quality.append(
                {
                    "target": name,
                    "condition": condition,
                    "candidates": CANDIDATES_PER_TARGET,
                    "ties": int(data["ties"].sum()),
                    "excluded": int(np.isnan(values[:, 0]).sum()),
                }
            )
            artifacts[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        combined = np.concatenate(blocks)
        if len(combined) != TARGETS_PER_CARDINALITY * CANDIDATES_PER_TARGET:
            raise ValueError(f"wrong sample count for {(events, condition)}")
        total_pairs += len(combined)
        for loss, name in enumerate(NAMES):
            valid = combined[:, loss][np.isfinite(combined[:, loss])]
            means[loss, column] = valid.mean()
            deviations[loss, column] = valid.std(ddof=1)
            rows.append(
                {
                    "metric": "phrase-cosine",
                    "events": events,
                    "condition": condition,
                    "loss": name,
                    "mean": means[loss, column],
                    "candidate_sd": deviations[loss, column],
                    "eligible": len(valid),
                    "targets": TARGETS_PER_CARDINALITY,
                    "candidates_per_target": CANDIDATES_PER_TARGET,
                }
            )
    expected = len(COLUMNS) * TARGETS_PER_CARDINALITY * CANDIDATES_PER_TARGET
    if total_pairs != expected or expected != 90_112:
        raise ValueError(f"wrong total pair count: {total_pairs}")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "phrase-cosine.csv", rows)
    write_csv(output / "quality.csv", quality)
    save_json(output / "design.json", design_metadata())
    save_json(output / "qualification.json", json.loads((root / "qualification.json").read_text()))
    headers = [
        f"{ {'joint': 'Joint', 'pitch': 'Known time', 'time': 'Known f0'}[kind] }: {events}"
        for events, kind in COLUMNS
    ]
    markdown = [
        "# Mean whole-phrase cosine ± SD",
        "",
        "| Loss | " + " | ".join(headers) + " |",
        "|---|" + "---:|" * len(headers),
    ]
    for loss, label in enumerate(TEXT_LABELS):
        cells = [
            f"{compact_cosine(means[loss, column])} ± "
            f"{compact_cosine(deviations[loss, column])}"
            for column in range(len(COLUMNS))
        ]
        markdown.append("| " + " | ".join((label, *cells)) + " |")
    (output / "phrase-cosine.md").write_text("\n".join(markdown) + "\n")
    render(
        output,
        "phrase-cosine",
        means,
        deviations,
        COLUMNS,
        cmap_name="RdBu",
        cropped_colorbar=True,
        compact_layout=True,
        labels=LABELS,
    )
    save_json(
        output / "provenance.json",
        {
            **design_metadata(),
            "source_hashes": signature()[1],
            "report_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "raw_artifacts": artifacts,
            "eligible_pairs_per_loss": total_pairs,
            "metric": "whole-phrase cosine after unrestricted Hungarian matching",
            "sd": "Across candidate phrases, ddof=1; descriptive, not a confidence interval",
            "renderer": PhraseSynth().provenance(),
            "execution": {
                f"{events}-{index:03d}": json.loads(
                    (root / f"execution-c{events}-t{index:03d}.json").read_text()
                )
                for events in CARDINALITIES
                for index in range(TARGETS_PER_CARDINALITY)
            },
        },
    )
    if paper_figure is not None:
        paper_figure.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output / "phrase-cosine.pdf", paper_figure)
    print(output / "phrase-cosine.png", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("qualify", "compute", "report", "all"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--paper-figure", type=Path, default=PAPER_FIGURE)
    parser.add_argument("--cardinality", type=int, choices=CARDINALITIES)
    parser.add_argument("--target-index", type=int)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    if args.command in ("qualify", "all"):
        qualify(args.root, args.device, args.batch)
    if args.command in ("compute", "all"):
        selected = CARDINALITIES if args.cardinality is None else (args.cardinality,)
        for events in selected:
            compute_cardinality(
                args.root, events, args.batch, args.device, args.target_index
            )
    if args.command in ("report", "all"):
        report(args.root, args.output, args.paper_figure)


if __name__ == "__main__":
    main()
