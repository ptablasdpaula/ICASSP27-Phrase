"""Evaluate Log-Weighed and fixed-decay CeL direction variants on Experiment I."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from assess_direction_variants import (
    compact,
    phrase_cosine,
    registry,
    sampling_counts,
    source_path,
)
from assess_gradients import checked_data, save_json
from extend_gradient_cardinality import EXTENDED_COLUMNS
from icassp27_phrase.gradient_assessment import fade_matrix
from icassp27_phrase.losses import CEL_DIRECTIONS, CumulativeEnergyDistance
from icassp27_phrase.runtime import configure_reproducibility, require_df2_backend
from icassp27_phrase.synth import PhraseSynth

ROOT = Path("results/log-decay-direction-gradient")
OUTPUT = Path("docs/gradient-assessment/log-decay-directions")
VARIANTS = ("forward_log", "four_log_decay", "forward_log_decay")
LABELS = ("Forward-time logCeL", "log-decCeL", "Forward-time log-decCeL")
FORWARD = (0, 1)


def source_signature() -> tuple[str, dict[str, str]]:
    paths = (
        Path(__file__),
        Path("scripts/assess_direction_variants.py"),
        Path("scripts/test_fading_diagonal.py"),
        Path("scripts/run_fading_recovery.py"),
        Path("src/gradient_assessment.py"),
        Path("src/losses.py"),
        Path("src/synth.py"),
        Path("src/exciter.py"),
        Path("src/waveguide.py"),
    )
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), hashes


class Objectives:
    """Share one power STFT across the three requested CeL variants."""

    def __init__(self, target: torch.Tensor):
        self.canonical = CumulativeEnergyDistance(target)
        power = self.canonical._power(target[None])
        self.mass = power.sum().detach()
        self.fade_frequency = fade_matrix(power.shape[-2], 4000 / 256, 1000).to(
            device=target.device, dtype=target.dtype
        )
        self.fade_time = fade_matrix(power.shape[-1], 64 / 4000, 1).to(
            device=target.device, dtype=target.dtype
        )
        self.fade_reference = self.fade_features(power).detach()

    def fade_features(self, power: torch.Tensor) -> torch.Tensor:
        result = []
        for time_reverse in (False, True):
            for frequency_reverse in (False, True):
                axes = tuple(
                    axis
                    for axis, reverse in (
                        (-2, frequency_reverse),
                        (-1, time_reverse),
                    )
                    if reverse
                )
                oriented = power.flip(axes) if axes else power
                surface = self.fade_frequency @ oriented @ self.fade_time.T
                surface = surface.flip(axes) if axes else surface
                result.append((surface / self.mass).clamp_min(1e-12).sqrt())
        return torch.stack(result, dim=1)

    def elementary(self, power: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_terms = []
        for index, (time_reverse, frequency_reverse) in enumerate(
            (tr, fr) for tr in (False, True) for fr in (False, True)
        ):
            surface = power.flip(-1).cumsum(-1).flip(-1) if time_reverse else power.cumsum(-1)
            surface = (
                surface.flip(-2).cumsum(-2).flip(-2)
                if frequency_reverse
                else surface.cumsum(-2)
            )
            reference, mass = self.canonical.references[index]
            error = (surface / mass).clamp_min(1e-12).sqrt() - reference
            log_terms.append(
                torch.linalg.vector_norm(
                    (error * self.canonical.sqrt_weights[index]).flatten(1), dim=1
                )
            )
        log_terms = torch.stack(log_terms, dim=1)
        fade_error = self.fade_features(power) - self.fade_reference
        log_decay_terms = torch.linalg.vector_norm(
            (fade_error * self.canonical.sqrt_weights).flatten(2), dim=-1
        )
        return log_terms, log_decay_terms

    def values(self, audio: torch.Tensor) -> torch.Tensor:
        log_terms, log_decay_terms = self.elementary(self.canonical._power(audio))
        return torch.stack(
            (
                log_terms[:, FORWARD].mean(1),
                log_decay_terms.mean(1),
                log_decay_terms[:, FORWARD].mean(1),
            ),
            dim=1,
        )

    def evaluate(self, synth: PhraseSynth, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coordinates = torch.tensor(
            positions, dtype=torch.float64, device=self.mass.device, requires_grad=True
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
            raise FloatingPointError("non-finite Log/decay direction objective or gradient")
        return value, gradient


def setup() -> None:
    torch.set_num_threads(1)
    configure_reproducibility()
    require_df2_backend("cuda")


def qualify(root: Path) -> None:
    setup()
    assert CEL_DIRECTIONS == ("right_up", "right_down", "left_up", "left_down")
    synth = PhraseSynth().to("cuda")
    name, target = registry()[1]
    source = checked_data(source_path(name, len(target), "joint", 512))
    positions = source["candidates"][:4]
    target_coordinates = torch.tensor(target, dtype=torch.float64, device="cuda")
    with torch.no_grad():
        target_audio = synth(
            80 * 2 ** target_coordinates[None, :, 0], target_coordinates[None, :, 1]
        )[0]
    objective = Objectives(target_audio)
    coordinates = torch.tensor(positions, dtype=torch.float64, device="cuda", requires_grad=True)
    audio = synth(80 * 2 ** coordinates[..., 0], coordinates[..., 1])
    log_terms, log_decay_terms = objective.elementary(objective.canonical._power(audio))
    established = objective.canonical.directional_distances(audio)[:, 1]
    torch.testing.assert_close(log_terms, established, rtol=1e-12, atol=1e-12)
    values = objective.values(audio)
    torch.testing.assert_close(values[:, 0], established[:, FORWARD].mean(1))
    torch.testing.assert_close(values[:, 1], log_decay_terms.mean(1))
    torch.testing.assert_close(values[:, 2], log_decay_terms[:, FORWARD].mean(1))

    # Match the already-qualified four-direction Log-Weighed fading implementation.
    from run_fading_recovery import FixedFading

    cpu_synth = PhraseSynth()
    cpu_target_coordinates = torch.tensor(target, dtype=torch.float64)
    with torch.no_grad():
        cpu_target_audio = cpu_synth(
            80 * 2 ** cpu_target_coordinates[None, :, 0], cpu_target_coordinates[None, :, 1]
        )[0]
    cpu_objective = Objectives(cpu_target_audio)
    cpu_coordinates = torch.tensor(positions, dtype=torch.float64, requires_grad=True)
    cpu_audio = cpu_synth(80 * 2 ** cpu_coordinates[..., 0], cpu_coordinates[..., 1])
    reference = FixedFading(cpu_target_audio, weighted=True)
    for row in range(len(cpu_audio)):
        torch.testing.assert_close(
            cpu_objective.values(cpu_audio[row : row + 1])[0, 1],
            reference(cpu_audio[row]),
            rtol=1e-13,
            atol=1e-14,
        )

    same = target_audio.clone().requires_grad_()
    self_values = objective.values(same[None])[0]
    assert self_values.abs().max() == 0
    self_gradient = torch.autograd.grad(self_values.sum(), same)[0]
    assert self_gradient.abs().max() == 0
    gradient = torch.autograd.grad(values.sum(), coordinates)[0]
    assert torch.isfinite(gradient).all() and gradient.norm() > 0

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
                "forward directions are exactly right_up and right_down",
                "unfaded Log-Weighed terms equal CumulativeEnergyDistance",
                "four-direction Log-Weighed decay equals established FixedFading",
                "requested subset means use explicit direction indices",
                "zero self-loss and self-gradient",
                "finite nonzero parameter gradients",
            ],
            "direction_order": list(CEL_DIRECTIONS),
            "forward_indices": list(FORWARD),
            "time_horizon_seconds": 1,
            "frequency_horizon_hz": 1000,
            "log_weighing": True,
        },
    )


def raw_path(root: Path, events: int, condition: str, name: str) -> Path:
    return root / "raw" / f"{condition}-{events}" / f"{name}.npz"


def compute(root: Path, batch: int) -> None:
    setup()
    qualification = json.loads((root / "qualification.json").read_text())
    experimental_signature = source_signature()[0]
    if not qualification["passed"] or qualification["signature"] != experimental_signature:
        raise ValueError("missing or stale Log/decay direction qualification")
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
                objective = Objectives(target_audio)
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


def report(root: Path, output: Path) -> None:
    execution = json.loads((root / "execution.json").read_text())
    if not execution["complete"] or execution["signature"] != source_signature()[0]:
        raise ValueError("Log/decay direction execution is incomplete or stale")
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
        "# Log-Weighed decay/direction CeL variants on Experiment I",
        "",
        "Whole-phrase cosine, reported as candidate mean ± sample SD. Frozen targets, ",
        "candidate phrases, Hungarian assignments and accepted sample counts are identical to ",
        "Experiment I. Forward-time variants average right/up and right/down. Decay is the fixed ",
        "1-second/1000-Hz kernel used by decCeL. Every row uses the existing Log-Weighing ",
        "quadrature; `log` does not alter the accumulated power or decay coordinates.",
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
                "forward_log": ["right_up", "right_down"],
                "four_log_decay": list(CEL_DIRECTIONS),
                "forward_log_decay": ["right_up", "right_down"],
            },
            "time_horizon_seconds": 1,
            "frequency_horizon_hz": 1000,
            "log_weighing": True,
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
