"""Evaluate three CeL direction variants on the frozen Experiment I design."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from assess_gradients import checked_data, save_json, shard_path
from extend_gradient_cardinality import EXTENDED_COLUMNS, extra_targets
from icassp27_phrase.gradient_assessment import reverse_cumsum, signature, targets
from icassp27_phrase.losses import CumulativeEnergyDistance
from icassp27_phrase.runtime import configure_reproducibility, require_df2_backend
from icassp27_phrase.synth import PhraseSynth
from screen_eight_directions import AxisLoss, axis_surfaces

BASE_ROOT = Path("results/gradient-assessment-gpu")
EXTENSION_ROOT = Path("results/gradient-assessment-extended")
ROOT = Path("results/direction-variant-gradient")
OUTPUT = Path("docs/gradient-assessment/direction-variants")
VARIANTS = ("orthogonal", "time_directions", "frequency_directions")
LABELS = ("Orthogonal CeL", "Time-direction CeL", "Frequency-direction CeL")


def source_signature() -> tuple[str, dict[str, str]]:
    paths = (
        Path(__file__),
        Path("scripts/screen_eight_directions.py"),
        Path("src/gradient_assessment.py"),
        Path("src/losses.py"),
        Path("src/synth.py"),
        Path("src/exciter.py"),
        Path("src/waveguide.py"),
    )
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), hashes


def registry() -> list[tuple[str, np.ndarray]]:
    rows = targets()
    rows.extend((name, target) for name, target in extra_targets() if len(target) > 4)
    return rows


def sampling_counts() -> tuple[dict[str, int], dict[str, int]]:
    base = json.loads((BASE_ROOT / "sampling.json").read_text())
    extension = json.loads((EXTENSION_ROOT / "sampling.json").read_text())
    expected = signature()[0]
    if not base["complete"] or not extension["complete"]:
        raise ValueError("Experiment I sampling is incomplete")
    if base["signature"] != expected or extension["signature"] != expected:
        raise ValueError("Experiment I sampling is stale")
    return base["final_counts"], extension["final_counts"]


def source_path(name: str, events: int, condition: str, count: int) -> Path:
    root = EXTENSION_ROOT if events in (6, 8) or events == 1 and condition != "joint" else BASE_ROOT
    return shard_path(root, name, condition, count, 0)


class DirectionObjectives:
    """Share one power STFT across diagonal-subset and orthogonal CeLs."""

    def __init__(self, target: torch.Tensor):
        self.canonical = CumulativeEnergyDistance(target)
        target_power = self.canonical._power(target[None])
        self.axis_mass = target_power.sum().clamp_min(torch.finfo(target.dtype).tiny)
        self.axis_reference = (
            (axis_surfaces(target_power) / self.axis_mass).clamp_min(1e-12).sqrt().detach()
        )

    def elementary(self, power: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        diagonal = []
        for index, (reverse_time, reverse_frequency) in enumerate(
            (tr, fr) for tr in (False, True) for fr in (False, True)
        ):
            surface = reverse_cumsum(power, 2) if reverse_time else power.cumsum(2)
            surface = reverse_cumsum(surface, 1) if reverse_frequency else surface.cumsum(1)
            reference, mass = self.canonical.references[index]
            error = (surface / mass).clamp_min(1e-12).sqrt() - reference
            diagonal.append(
                torch.linalg.vector_norm(error.flatten(1), dim=1)
                / math.sqrt(error.shape[-2] * error.shape[-1])
            )
        diagonal = torch.stack(diagonal, dim=1)
        axis_error = (
            (axis_surfaces(power) / self.axis_mass).clamp_min(1e-12).sqrt()
            - self.axis_reference
        )
        orthogonal = torch.linalg.vector_norm(axis_error.flatten(2), dim=-1) / math.sqrt(
            power.shape[-2] * power.shape[-1]
        )
        return diagonal, orthogonal

    def values(self, audio: torch.Tensor) -> torch.Tensor:
        diagonal, orthogonal = self.elementary(self.canonical._power(audio))
        # Diagonal order: right/up, right/down, left/up, left/down.
        return torch.stack(
            (
                orthogonal.mean(1),
                diagonal[:, (0, 1)].mean(1),
                diagonal[:, (1, 3)].mean(1),
            ),
            dim=1,
        )

    def evaluate(self, synth: PhraseSynth, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coordinates = torch.tensor(
            positions, dtype=torch.float64, device=self.axis_mass.device, requires_grad=True
        )
        audio = synth(80 * 2 ** coordinates[..., 0], coordinates[..., 1])
        values = self.values(audio)
        gradients = []
        for index in range(len(VARIANTS)):
            (gradient,) = torch.autograd.grad(
                values[:, index].sum(), coordinates, retain_graph=index < len(VARIANTS) - 1
            )
            gradients.append(gradient.detach())
        value = values.detach().cpu().numpy()
        gradient = torch.stack(gradients, dim=1).cpu().numpy()
        if not np.isfinite(value).all() or not np.isfinite(gradient).all():
            raise FloatingPointError("non-finite direction-variant objective or gradient")
        return value, gradient


def setup() -> None:
    torch.set_num_threads(1)
    configure_reproducibility()
    require_df2_backend("cuda")


def qualify(root: Path) -> None:
    setup()
    synth = PhraseSynth().to("cuda")
    name, target = targets()[1]
    source = checked_data(source_path(name, len(target), "joint", 512))
    positions = source["candidates"][:4]
    target_coordinates = torch.tensor(target, dtype=torch.float64, device="cuda")
    with torch.no_grad():
        target_audio = synth(
            80 * 2 ** target_coordinates[None, :, 0], target_coordinates[None, :, 1]
        )[0]
    objective = DirectionObjectives(target_audio)
    coordinates = torch.tensor(positions, dtype=torch.float64, device="cuda", requires_grad=True)
    audio = synth(80 * 2 ** coordinates[..., 0], coordinates[..., 1])
    power = objective.canonical._power(audio)
    diagonal, orthogonal = objective.elementary(power)
    diagonal_reference = objective.canonical.directional_distances(audio)[:, 0]
    axis_reference = AxisLoss(objective.canonical).terms(audio)[:, :4]
    torch.testing.assert_close(diagonal, diagonal_reference, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(orthogonal, axis_reference, rtol=1e-12, atol=1e-12)
    if torch.allclose(diagonal, orthogonal, rtol=1e-5, atol=1e-7):
        raise AssertionError("orthogonal terms unexpectedly equal diagonal terms")
    diagonal_gradient = torch.autograd.grad(diagonal.mean(), coordinates, retain_graph=True)[0]
    orthogonal_gradient = torch.autograd.grad(orthogonal.mean(), coordinates)[0]
    if torch.allclose(diagonal_gradient, orthogonal_gradient, rtol=1e-5, atol=1e-7):
        raise AssertionError("orthogonal gradients unexpectedly equal diagonal gradients")
    experimental_signature, hashes = source_signature()
    save_json(
        root / "qualification.json",
        {
            "passed": True,
            "device": "cuda",
            "gpu": torch.cuda.get_device_name(),
            "signature": experimental_signature,
            "source_hashes": hashes,
            "checks": [
                "diagonal elementary values equal CumulativeEnergyDistance",
                "orthogonal elementary values equal the established AxisLoss",
                "orthogonal values differ from diagonal values",
                "orthogonal gradients differ from diagonal gradients",
            ],
            "direction_order": {
                "diagonal": ["right_up", "right_down", "left_up", "left_down"],
                "orthogonal": ["up", "down", "right", "left"],
            },
        },
    )


def raw_path(root: Path, events: int, condition: str, name: str) -> Path:
    return root / "raw" / f"{condition}-{events}" / f"{name}.npz"


def compute(root: Path, batch: int) -> None:
    setup()
    qualification = json.loads((root / "qualification.json").read_text())
    experimental_signature = source_signature()[0]
    if not qualification["passed"] or qualification["signature"] != experimental_signature:
        raise ValueError("missing or stale direction-variant qualification")
    base_counts, extension_counts = sampling_counts()
    target_registry = registry()
    synth = PhraseSynth().to("cuda")
    started = time.perf_counter()
    total = 0
    for events, condition in EXTENDED_COLUMNS:
        key = f"{events}-{condition}"
        count = (extension_counts if key in extension_counts else base_counts)[key]
        for name, target in target_registry:
            if len(target) != events:
                continue
            source_file = source_path(name, events, condition, count)
            source = checked_data(source_file)
            output = raw_path(root, events, condition, name)
            source_sha256 = hashlib.sha256(source_file.read_bytes()).hexdigest()
            if output.exists():
                with np.load(output) as saved:
                    if (
                        str(saved["signature"]) != experimental_signature
                        or str(saved["source_sha256"]) != source_sha256
                    ):
                        raise ValueError(f"stale checkpoint: {output}")
                continue
            target_coordinates = torch.tensor(target, dtype=torch.float64, device="cuda")
            with torch.no_grad():
                target_audio = synth(
                    80 * 2 ** target_coordinates[None, :, 0], target_coordinates[None, :, 1]
                )[0]
                objective = DirectionObjectives(target_audio)
            values, gradients = [], []
            task_started = time.perf_counter()
            for begin in range(0, count, batch):
                value, gradient = objective.evaluate(
                    synth, source["candidates"][begin : begin + batch]
                )
                values.append(value)
                gradients.append(gradient)
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary,
                signature=experimental_signature,
                source_sha256=source_sha256,
                source_path=str(source_file),
                variants=np.array(VARIANTS),
                target=source["target"],
                candidates=source["candidates"],
                assignments=source["assignments"],
                ties=source["ties"],
                losses=np.concatenate(values),
                gradients=np.concatenate(gradients),
                device="cuda",
                gpu=torch.cuda.get_device_name(),
                wall_seconds=time.perf_counter() - task_started,
            )
            temporary.replace(output)
            total += count
            print(
                f"DONE {name} {condition} n={count} "
                f"{time.perf_counter() - task_started:.1f}s",
                flush=True,
            )
    save_json(
        root / "execution.json",
        {
            "complete": True,
            "signature": experimental_signature,
            "seconds": time.perf_counter() - started,
            "new_comparisons": total,
            "batch": batch,
            "device": "cuda",
            "gpu": torch.cuda.get_device_name(),
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            "job_id": __import__("os").environ.get("SLURM_JOB_ID", "manual"),
        },
    )


def checked_variant(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as saved:
        data = dict(saved)
    if str(data["signature"]) != source_signature()[0]:
        raise ValueError(f"stale result: {path}")
    count, events, dimensions = data["candidates"].shape
    if (
        dimensions != 2
        or tuple(data["variants"]) != VARIANTS
        or data["gradients"].shape != (count, len(VARIANTS), events, 2)
        or data["losses"].shape != (count, len(VARIANTS))
    ):
        raise ValueError(f"malformed result: {path}")
    if not np.isfinite(data["losses"]).all() or not np.isfinite(data["gradients"]).all():
        raise FloatingPointError(path)
    return data


def phrase_cosine(data: dict[str, np.ndarray]) -> np.ndarray:
    descent = -data["gradients"]
    delta = (data["target"][data["assignments"]] - data["candidates"])[:, None]
    dot = (descent * delta).sum(axis=(-2, -1))
    norms = np.linalg.norm(descent.reshape(*descent.shape[:2], -1), axis=-1)
    norms *= np.linalg.norm(delta.reshape(len(delta), -1), axis=-1)[:, None]
    cosine = np.divide(dot, norms, out=np.zeros_like(dot), where=norms > 0)
    cosine = np.clip(cosine, -1, 1)
    valid = ~data["ties"] & (np.abs(delta[:, 0]) > 1e-12).any(axis=(-2, -1))
    cosine[~valid] = np.nan
    return cosine


def compact(value: float) -> str:
    return f"{value:z.2f}".replace("0.", ".", 1) if abs(value) < 0.995 else f"{value:.2f}"


def report(root: Path, output: Path) -> None:
    execution = json.loads((root / "execution.json").read_text())
    if not execution["complete"] or execution["signature"] != source_signature()[0]:
        raise ValueError("direction-variant execution is incomplete or stale")
    base_counts, extension_counts = sampling_counts()
    target_registry = registry()
    rows = []
    matrix = np.zeros((len(VARIANTS), len(EXTENDED_COLUMNS)))
    deviations = np.zeros_like(matrix)
    eligible = np.zeros_like(matrix, dtype=int)
    raw_hashes = {}
    for column, (events, condition) in enumerate(EXTENDED_COLUMNS):
        key = f"{events}-{condition}"
        count = (extension_counts if key in extension_counts else base_counts)[key]
        blocks = []
        for name, target in target_registry:
            if len(target) != events:
                continue
            path = raw_path(root, events, condition, name)
            data = checked_variant(path)
            np.testing.assert_array_equal(data["target"], target)
            blocks.append(phrase_cosine(data))
            raw_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        values = np.concatenate(blocks)
        for variant, name in enumerate(VARIANTS):
            valid = values[:, variant][np.isfinite(values[:, variant])]
            mean, sd = float(valid.mean()), float(valid.std(ddof=1))
            matrix[variant, column] = mean
            deviations[variant, column] = sd
            eligible[variant, column] = len(valid)
            rows.append(
                {
                    "events": events,
                    "condition": condition,
                    "loss": name,
                    "mean_phrase_cosine": mean,
                    "candidate_sd": sd,
                    "eligible": len(valid),
                    "targets": len(blocks),
                    "candidates_per_target": count,
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    headers = [
        f"{ {'joint': 'Joint', 'pitch': 'Pitch', 'time': 'Time'}[condition] } {events}"
        for events, condition in EXTENDED_COLUMNS
    ]
    lines = [
        "# CeL direction variants on Experiment I",
        "",
        "Whole-phrase cosine, reported as candidate mean ± sample SD. The frozen targets, ",
        "candidate phrases, Hungarian assignments and accepted sample counts are identical to ",
        "Experiment I. Orthogonal CeL averages ↑, ↓, → and ← one-axis cumulative sums; Time ",
        "directions average ↗ and ↘; Frequency directions average ↘ and ↙. All variants use ",
        "uniform STFT-bin weighting and no decay.",
        "",
        "| Loss | " + " | ".join(headers) + " |",
        "|---|" + "---:|" * len(headers),
    ]
    for index, label in enumerate(LABELS):
        cells = [
            f"{compact(matrix[index, column])} ± {compact(deviations[index, column])}"
            for column in range(len(headers))
        ]
        lines.append("| " + " | ".join((label, *cells)) + " |")
    (output / "README.md").write_text("\n".join(lines) + "\n")
    save_json(
        output / "provenance.json",
        {
            "signature": source_signature()[0],
            "source_hashes": source_signature()[1],
            "raw_hashes": raw_hashes,
            "columns": EXTENDED_COLUMNS,
            "variants": {
                "orthogonal": ["up", "down", "right", "left"],
                "time_directions": ["right_up", "right_down"],
                "frequency_directions": ["right_down", "left_down"],
            },
            "metric": "whole-phrase cosine after unrestricted Hungarian matching",
            "sd": "Across candidate phrases, ddof=1; descriptive, not a confidence interval",
            "total_comparisons": int(eligible.sum() // len(VARIANTS)),
            "execution": execution,
        },
    )
    print(output / "README.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("qualify", "compute", "report", "all"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    if args.command in ("qualify", "all"):
        qualify(args.root)
    if args.command in ("compute", "all"):
        compute(args.root, args.batch)
    if args.command in ("report", "all"):
        report(args.root, args.output)


if __name__ == "__main__":
    main()
