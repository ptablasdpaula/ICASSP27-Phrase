#!/usr/bin/env python3
"""Assess persistent-target mismatch and single-event all-control gradients."""

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
from icassp27_phrase.data.design import (
    CANDIDATES_PER_TARGET,
    NAMES,
    TARGETS_PER_CARDINALITY,
    FixedObjectives,
    all_control_candidates,
    all_control_targets,
    candidates,
    persistent_targets,
)
from icassp27_phrase.metrics import hungarian_assignment, phrase_gradient_cosine
from icassp27_phrase.paths import OUTPUT as OUTPUT_ROOT
from icassp27_phrase.paths import resolve_path
from icassp27_phrase.runtime import configure_reproducibility, require_df2_backend
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.synth.config import ExciterConfig, WaveguideConfig
from icassp27_phrase.synth.controls import (
    CONTROL_NAMES,
    CONTROL_RANGES,
    decode_controls,
    dynamic_excitation,
    dynamic_transfer_rows,
    encode_defaults,
    render_all_controls,
)
from icassp27_phrase.synth.waveguide import (
    _df2_filter,
)

ROOT = OUTPUT_ROOT / "gradient-limitations"
OUTPUT = OUTPUT_ROOT / "gradient-limitations"
SCHEMA = "gradient-assessment-limitations-v1"
CONDITIONS = ("persistent_target", "all_controls")

# Broad ranges inherited where possible from the earlier plucked-string model.
# Positive scale controls use logarithmic unit coordinates, and loop gain uses
# log(1-g), matching its multiplicative effect on decay.


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def setup(device: str) -> None:
    torch.set_num_threads(1)
    configure_reproducibility()
    require_df2_backend(device)


class AllControlObjectives(FixedObjectives):
    def evaluate(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coordinates = torch.tensor(
            positions,
            dtype=torch.float64,
            device=self.target.device,
            requires_grad=True,
        )
        terms = self.values(render_all_controls(coordinates))
        values, gradients = [], []
        for index, name in enumerate(NAMES):
            (gradient,) = torch.autograd.grad(
                terms[name].sum(), coordinates, retain_graph=index < len(NAMES) - 1
            )
            values.append(terms[name].detach())
            gradients.append(gradient.detach())
        value = torch.stack(values, 1).cpu().numpy()
        gradient = torch.stack(gradients, 1).cpu().numpy()
        if not np.isfinite(value).all() or not np.isfinite(gradient).all():
            raise FloatingPointError("nonfinite all-control objective/gradient")
        return value, gradient


def signature():
    from icassp27_phrase.paths import scientific_signature

    return scientific_signature()


def raw_path(root: Path, condition: str, target_index: int) -> Path:
    return root / "raw" / condition / f"target-{target_index:03d}.npz"


def qualify(root: Path, device: str) -> None:
    setup(device)
    dynamic_unit = encode_defaults(device=device).requires_grad_()
    # The logarithmic round trip through the unit coordinate can place the
    # nominal 10-ms duration a few ulps above an integer sample count, changing
    # its hard support by one sample. Use the exact fixed controls for this
    # renderer-equivalence gate; random LHS durations almost surely avoid that
    # measure-zero boundary.
    fixed_controls = decode_controls(dynamic_unit)
    fixed_controls["duration_seconds"] = torch.full_like(
        fixed_controls["duration_seconds"], ExciterConfig().duration_seconds
    )
    dynamic_source = dynamic_excitation(fixed_controls)
    dynamic_numerator, dynamic_denominator = dynamic_transfer_rows(
        fixed_controls, WaveguideConfig()
    )
    dynamic_audio = _df2_filter(dynamic_source, dynamic_numerator, dynamic_denominator)
    fixed_synth = PhraseSynth().to(device)
    fixed_coordinate = torch.tensor([[160.0]], dtype=torch.float64, device=device)
    fixed_onset = torch.tensor([[1.0]], dtype=torch.float64, device=device)
    with torch.no_grad():
        fixed_audio = fixed_synth(fixed_coordinate, fixed_onset)
    difference = (dynamic_audio.detach() - fixed_audio).abs()
    torch.testing.assert_close(dynamic_audio.detach(), fixed_audio, rtol=2e-9, atol=2e-10)
    probe = dynamic_unit.detach().clone()
    probe[0] = torch.tensor(
        [0.43, 0.37, 0.58, 0.41, 0.39, 0.55, 0.31],
        dtype=torch.float64,
        device=device,
    )
    probe.requires_grad_()
    rendered = render_all_controls(probe)
    weights = torch.linspace(-1.0, 1.0, rendered.shape[-1], device=device)
    gradient = torch.autograd.grad((rendered * weights).sum(), probe)[0]
    if not bool(torch.isfinite(gradient).all()) or bool((gradient == 0.0).any()):
        raise FloatingPointError("one or more all-control derivatives are zero/nonfinite")
    persistent = PhraseSynth(
        waveguide_config=replace(WaveguideConfig(), state_policy="persistent")
    ).to(device)
    target = persistent_targets()[0][1]
    target_tensor = torch.tensor(target, dtype=torch.float64, device=device)
    with torch.no_grad():
        persistent_audio = persistent(
            80.0 * 2.0 ** target_tensor[None, :, 0], target_tensor[None, :, 1]
        )
    if not bool(torch.isfinite(persistent_audio).all()):
        raise FloatingPointError("persistent target renderer failed")
    save_json(
        root / "qualification.json",
        {
            "passed": True,
            "signature": signature()[0],
            "source_hashes": signature()[1],
            "device": device,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
            "default_renderer_max_abs_error": float(difference.max().cpu()),
            "default_renderer_rms_error": float(difference.square().mean().sqrt().cpu()),
            "all_control_probe_gradient": gradient.detach().cpu().tolist()[0],
            "control_names": list(CONTROL_NAMES),
            "control_ranges": CONTROL_RANGES,
        },
    )
    print("QUALIFIED", root / "qualification.json", flush=True)


def _save_shard(path: Path, **values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, schema=SCHEMA, signature=signature()[0], **values)
    temporary.replace(path)


def completed(root: Path, condition: str, index: int) -> bool:
    qualification = json.loads((root / "qualification.json").read_text())
    if not qualification["passed"] or qualification["signature"] != signature()[0]:
        raise ValueError("Run gradients qualify: qualification is missing or stale")
    path = raw_path(root, condition, index)
    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as saved:
        expected_shape = (
            (256, len(NAMES), 2, 2) if condition == "persistent_target" else (256, len(NAMES), 7)
        )
        if (
            str(saved["signature"]) != signature()[0]
            or str(saved["schema"]) != SCHEMA
            or tuple(saved["names"]) != NAMES
            or saved["gradients"].shape != expected_shape
            or saved["losses"].shape != (256, len(NAMES))
            or not np.isfinite(saved["gradients"]).all()
            or not np.isfinite(saved["losses"]).all()
        ):
            raise ValueError(f"Incomplete or stale gradient shard: {path}")
    return True


def compute_persistent(root: Path, target_index: int, batch: int, device: str) -> None:
    if completed(root, "persistent_target", target_index):
        return
    setup(device)
    name, target = persistent_targets()[target_index]
    positions = candidates(name, target)["joint"]
    target_coordinates = torch.tensor(target, dtype=torch.float64, device=device)
    target_synth = PhraseSynth(
        waveguide_config=replace(WaveguideConfig(), state_policy="persistent")
    ).to(device)
    candidate_synth = PhraseSynth().to(device)
    with torch.no_grad():
        target_audio = target_synth(
            80.0 * 2.0 ** target_coordinates[None, :, 0],
            target_coordinates[None, :, 1],
        )[0]
        objective = FixedObjectives(target_audio)
    values, gradients = [], []
    started = time.perf_counter()
    for begin in range(0, len(positions), batch):
        value, gradient = objective.evaluate(candidate_synth, positions[begin : begin + batch])
        values.append(value)
        gradients.append(gradient)
    assignments, ties = zip(*(hungarian_assignment(row, target) for row in positions), strict=True)
    _save_shard(
        raw_path(root, "persistent_target", target_index),
        condition="persistent_target",
        target_id=name,
        target=target,
        candidates=positions,
        losses=np.concatenate(values),
        gradients=np.concatenate(gradients),
        assignments=np.stack(assignments),
        ties=np.asarray(ties),
        names=np.asarray(NAMES),
        target_renderer_json=json.dumps(target_synth.provenance(), sort_keys=True),
        candidate_renderer_json=json.dumps(candidate_synth.provenance(), sort_keys=True),
        wall_seconds=time.perf_counter() - started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    print(f"DONE persistent_target {target_index:03d}", flush=True)


def compute_all_controls(root: Path, target_index: int, batch: int, device: str) -> None:
    if completed(root, "all_controls", target_index):
        return
    setup(device)
    name, target = all_control_targets()[target_index]
    positions = all_control_candidates(name)
    target_tensor = torch.tensor(target[None], dtype=torch.float64, device=device)
    with torch.no_grad():
        target_audio = render_all_controls(target_tensor)[0]
        objective = AllControlObjectives(target_audio)
    values, gradients = [], []
    started = time.perf_counter()
    for begin in range(0, len(positions), batch):
        value, gradient = objective.evaluate(positions[begin : begin + batch])
        values.append(value)
        gradients.append(gradient)
    _save_shard(
        raw_path(root, "all_controls", target_index),
        condition="all_controls",
        target_id=name,
        target=target,
        candidates=positions,
        losses=np.concatenate(values),
        gradients=np.concatenate(gradients),
        names=np.asarray(NAMES),
        control_names=np.asarray(CONTROL_NAMES),
        control_ranges_json=json.dumps(CONTROL_RANGES, sort_keys=True),
        wall_seconds=time.perf_counter() - started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    print(f"DONE all_controls {target_index:03d}", flush=True)


def cosine(gradient: np.ndarray, candidate: np.ndarray, target: np.ndarray) -> np.ndarray:
    displacement = target[None, None] - candidate[:, None]
    descent = -gradient
    dot = (descent * displacement).sum(axis=-1)
    norm = np.linalg.norm(descent, axis=-1) * np.linalg.norm(displacement, axis=-1)
    return np.clip(np.divide(dot, norm, out=np.zeros_like(dot), where=norm > 0), -1, 1)


def report(root: Path, output: Path) -> None:
    collected = {"persistent_target": [], "all_controls": []}
    artifacts: dict[str, str] = {}
    for index, (_, target) in enumerate(persistent_targets()):
        path = raw_path(root, "persistent_target", index)
        with np.load(path) as saved:
            data = dict(saved)
        if str(data["schema"]) != SCHEMA or str(data["signature"]) != signature()[0]:
            raise ValueError(f"stale persistent shard: {path}")
        values = phrase_gradient_cosine(
            data["gradients"],
            data["candidates"],
            target,
            data["assignments"],
            data["ties"],
        )
        collected["persistent_target"].append(values)
        artifacts[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for index, (_, target) in enumerate(all_control_targets()):
        path = raw_path(root, "all_controls", index)
        with np.load(path) as saved:
            data = dict(saved)
        if str(data["schema"]) != SCHEMA or str(data["signature"]) != signature()[0]:
            raise ValueError(f"stale all-control shard: {path}")
        collected["all_controls"].append(cosine(data["gradients"], data["candidates"], target))
        artifacts[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    combined = {key: np.concatenate(blocks) for key, blocks in collected.items()}
    expected = TARGETS_PER_CARDINALITY * CANDIDATES_PER_TARGET
    if any(value.shape != (expected, len(NAMES)) for value in combined.values()):
        raise ValueError("one or more limitation columns has the wrong sample count")
    rows = []
    for loss_index, loss in enumerate(NAMES):
        for condition, values in combined.items():
            selected = values[:, loss_index]
            selected = selected[np.isfinite(selected)]
            rows.append(
                {
                    "loss": loss,
                    "condition": condition,
                    "mean": float(selected.mean()),
                    "sample_sd": float(selected.std(ddof=1)),
                    "eligible": len(selected),
                }
            )
    write_csv(output / "summary.csv", rows)
    by_key = {(row["loss"], row["condition"]): row for row in rows}
    labels = {
        "persistent_target": "Persistent target, N=2",
        "all_controls": "All seven controls, N=1",
    }
    markdown = [
        "# Additional gradient diagnostics",
        "",
        "Mean phrase-wide cosine ± sample SD.",
        "",
        "| Loss | " + " | ".join(labels.values()) + " |",
        "|---|" + "---:|" * len(labels),
    ]
    display = dict(
        zip(
            NAMES,
            (
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
            ),
            strict=True,
        )
    )
    for loss in NAMES:
        cells = []
        for condition in labels:
            row = by_key[(loss, condition)]
            cells.append(
                f"{row['mean']:.3f} ± {row['sample_sd']:.3f} ({row['positive_percent']:.1f}%)"
            )
        markdown.append("| " + " | ".join((display[loss], *cells)) + " |")
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.md").write_text("\n".join(markdown) + "\n")
    save_json(
        output / "provenance.json",
        {
            "schema": SCHEMA,
            "signature": signature()[0],
            "source_hashes": signature()[1],
            "targets_per_condition": TARGETS_PER_CARDINALITY,
            "candidates_per_target": CANDIDATES_PER_TARGET,
            "pairs_per_condition": expected,
            "control_names": list(CONTROL_NAMES),
            "control_ranges": CONTROL_RANGES,
            "coordinate_system": (
                "Each all-control coordinate spans [0,1]; f0, amplitude, and duration "
                "are logarithmic, loop gain is logarithmic in 1-g, and the remaining "
                "coordinates are linear."
            ),
            "persistent_target": (
                "N=2 target uses persistent DF2 state; candidate uses the registered "
                "hard-reset renderer; pitch/onset LHS and Hungarian metric are unchanged."
            ),
            "all_controls": (
                "N=1 independent seven-dimensional scrambled LHS targets and candidates."
            ),
            "artifacts": artifacts,
        },
    )
    print(output / "summary.md", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("qualify", "compute", "report"))
    parser.add_argument("--root", type=resolve_path, default=ROOT)
    parser.add_argument("--output", type=resolve_path, default=OUTPUT)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--target-index", type=int)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    if args.command == "qualify":
        qualify(args.root, args.device)
    elif args.command == "compute":
        if args.condition is None or args.target_index is None:
            parser.error("compute requires --condition and --target-index")
        if not 0 <= args.target_index < TARGETS_PER_CARDINALITY:
            parser.error("--target-index must lie in [0,31]")
        qualification = json.loads((args.root / "qualification.json").read_text())
        if not qualification["passed"] or qualification["signature"] != signature()[0]:
            raise ValueError("qualification is missing or stale")
        if args.condition == "persistent_target":
            compute_persistent(args.root, args.target_index, args.batch, args.device)
        else:
            compute_all_controls(args.root, args.target_index, args.batch, args.device)
    else:
        report(args.root, args.output)


if __name__ == "__main__":
    main()
