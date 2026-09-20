#!/usr/bin/env python3
"""Assess persistent-target mismatch and single-event all-control gradients."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import resource
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc
from torch import Tensor

from fixed_gradient_assessment import (
    CANDIDATES_PER_TARGET,
    NAMES,
    TARGETS_PER_CARDINALITY,
    FixedObjectives,
    candidates,
    seed_for,
    targets,
)
from icassp27_phrase._fractional import (
    hermitian_rfft_projection,
    thiran_anchor,
)
from icassp27_phrase.config import ExciterConfig, WaveguideConfig
from icassp27_phrase.metrics import hungarian_assignment, phrase_gradient_cosine
from icassp27_phrase.runtime import configure_reproducibility, require_df2_backend
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.waveguide import (
    _RationalRows,
    _delay_rows,
    _df2_filter,
    _rational_add,
    _rational_multiply,
    _rational_scale,
    _unwrapped_phase,
)

ROOT = Path("results/gradient-assessment-limitations")
OUTPUT = Path("docs/gradient-assessment/limitations")
SCHEMA = "gradient-assessment-limitations-v1"
CONDITIONS = ("persistent_target", "all_controls")
CONTROL_NAMES = (
    "log_f0",
    "onset",
    "log_amplitude",
    "log_duration",
    "pluck_position",
    "log_one_minus_gain",
    "loop_pole",
)
TIMBRE_INDICES = np.arange(2, len(CONTROL_NAMES))

# Broad ranges inherited where possible from the earlier plucked-string model.
# Positive scale controls use logarithmic unit coordinates, and loop gain uses
# log(1-g), matching its multiplicative effect on decay.
CONTROL_RANGES = {
    "f0_hz": (80.0, 320.0),
    "onset_seconds": (0.2, 1.8),
    "amplitude": (0.1, 1.0),
    "duration_seconds": (0.0025, 0.040),
    # Thiran-3 requires both physical rail sections to retain at least its
    # stable fractional-delay support at the highest registered pitch.
    "pluck_position": (0.10, 0.50),
    "loop_gain": (0.90, 0.9999),
    "loop_pole": (0.0, 0.75),
}


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


def persistent_targets() -> list[tuple[str, np.ndarray]]:
    return [(name, value) for name, value in targets() if len(value) == 2]


def all_control_targets() -> list[tuple[str, np.ndarray]]:
    design = qmc.LatinHypercube(
        d=len(CONTROL_NAMES),
        scramble=True,
        optimization=None,
        seed=seed_for("limitations", "all-control-targets"),
    ).random(TARGETS_PER_CARDINALITY)
    return [(f"A01-T{index:03d}", row) for index, row in enumerate(design)]


def all_control_candidates(name: str) -> np.ndarray:
    return qmc.LatinHypercube(
        d=len(CONTROL_NAMES),
        scramble=True,
        optimization=None,
        seed=seed_for("limitations", "all-control-candidates", name),
    ).random(CANDIDATES_PER_TARGET)


def _log_scale(unit: Tensor, lower: float, upper: float) -> Tensor:
    return torch.exp(math.log(lower) + unit * math.log(upper / lower))


def decode_controls(unit: Tensor) -> dict[str, Tensor]:
    if unit.ndim != 2 or unit.shape[1] != len(CONTROL_NAMES):
        raise ValueError("all-control coordinates must have shape [batch,7]")
    if not bool(((unit.detach() >= 0.0) & (unit.detach() <= 1.0)).all()):
        raise ValueError("all-control coordinates must lie in [0,1]")
    gain_lower, gain_upper = CONTROL_RANGES["loop_gain"]
    one_minus_gain = _log_scale(unit[:, 5], 1.0 - gain_lower, 1.0 - gain_upper)
    return {
        "f0_hz": 80.0 * torch.pow(2.0, 2.0 * unit[:, 0]),
        "onset_seconds": 0.2 + 1.6 * unit[:, 1],
        "amplitude": _log_scale(unit[:, 2], *CONTROL_RANGES["amplitude"]),
        "duration_seconds": _log_scale(
            unit[:, 3], *CONTROL_RANGES["duration_seconds"]
        ),
        "pluck_position": 0.10 + 0.40 * unit[:, 4],
        "loop_gain": 1.0 - one_minus_gain,
        "loop_pole": 0.75 * unit[:, 6],
    }


def encode_defaults(*, device: torch.device | str) -> Tensor:
    exciter, waveguide = ExciterConfig(), WaveguideConfig()

    def log_unit(value: float, lower: float, upper: float) -> float:
        return math.log(value / lower) / math.log(upper / lower)

    gain_lower, gain_upper = CONTROL_RANGES["loop_gain"]
    values = [
        math.log2(160.0 / 80.0) / 2.0,
        (1.0 - 0.2) / 1.6,
        log_unit(exciter.amplitude, *CONTROL_RANGES["amplitude"]),
        log_unit(exciter.duration_seconds, *CONTROL_RANGES["duration_seconds"]),
        (waveguide.pluck_position - 0.10) / 0.40,
        log_unit(1.0 - waveguide.loop_gain, 1.0 - gain_lower, 1.0 - gain_upper),
        waveguide.loop_pole / 0.75,
    ]
    return torch.tensor(values, dtype=torch.float64, device=device)[None]


def dynamic_excitation(controls: dict[str, Tensor]) -> Tensor:
    config = ExciterConfig()
    reference = controls["f0_hz"]
    sample = torch.arange(
        config.sample_count, dtype=torch.float64, device=reference.device
    )[None]
    width = controls["duration_seconds"][:, None] * float(config.sample_rate)
    support = (sample < width.detach()).to(reference.dtype)
    prototype = (
        controls["amplitude"][:, None]
        * 0.5
        * (1.0 - torch.cos(torch.pi * sample / width))
        * support
    )
    spectrum = torch.fft.rfft(prototype, n=config.fourier_fft_length, dim=-1)
    omega = (
        2.0
        * torch.pi
        * torch.arange(
            config.fourier_fft_length // 2 + 1,
            dtype=reference.dtype,
            device=reference.device,
        )
        / float(config.fourier_fft_length)
    )
    phase = torch.exp(
        torch.complex(
            torch.zeros_like(controls["onset_seconds"][:, None] * omega[None]),
            -controls["onset_seconds"][:, None]
            * float(config.sample_rate)
            * omega[None],
        )
    )
    shifted = hermitian_rfft_projection(spectrum * phase, config.fourier_fft_length)
    return torch.fft.irfft(
        shifted, n=config.fourier_fft_length, dim=-1
    )[:, : config.sample_count]


def dynamic_transfer_rows(
    controls: dict[str, Tensor], config: WaveguideConfig
) -> tuple[Tensor, Tensor]:
    """Tensor-valued counterpart of the paper's fixed-config transfer builder."""
    frequency = controls["f0_hz"].reshape(-1)
    pole = controls["loop_pole"].reshape(-1)
    gain = controls["loop_gain"].reshape(-1)
    beta = controls["pluck_position"].reshape(-1)
    omega = 2.0 * math.pi * frequency / float(config.sample_rate)
    loop_phase = -torch.atan2(
        pole * torch.sin(omega), 1.0 - pole * torch.cos(omega)
    )
    rail = 0.5 * (float(config.sample_rate) / frequency + loop_phase / omega)
    initial_long = (1.0 - beta) * rail
    initial_short = beta * rail
    long_order, short_order = config.segment_orders or (0, 0)
    if config.interpolation != "thiran" or (long_order, short_order) != (3, 3):
        raise ValueError("the all-control diagnostic requires the paper's Thiran-3 path")
    long_whole = thiran_anchor(initial_long, long_order)
    short_whole = thiran_anchor(initial_short, short_order)
    for _ in range(config.phase_correction_iterations):
        long_phase = _unwrapped_phase(
            (1.0 - beta) * rail,
            config.interpolation,
            long_order,
            long_whole,
            omega,
        )
        short_phase = _unwrapped_phase(
            beta * rail,
            config.interpolation,
            short_order,
            short_whole,
            omega,
        )
        residual = 2.0 * (long_phase + short_phase) + loop_phase + 2.0 * math.pi
        rail = rail + residual / (2.0 * omega)
    long = _delay_rows(
        (1.0 - beta) * rail, config.interpolation, long_order, long_whole
    )
    short = _delay_rows(beta * rail, config.interpolation, short_order, short_whole)
    loop = _RationalRows(
        (gain * (1.0 - pole))[:, None],
        torch.stack((torch.ones_like(pole), -pole), dim=1),
    )
    rail_filter = _rational_multiply(long, short)
    feedback = _rational_multiply(
        loop, _rational_multiply(rail_filter, rail_filter)
    )
    identity = _RationalRows(
        torch.ones_like(frequency[:, None]), torch.ones_like(frequency[:, None])
    )
    characteristic = _rational_add(identity, feedback, right_scale=-1.0)
    closed_loop = _RationalRows(
        characteristic.denominator, characteristic.numerator
    )
    nut_comb = _rational_add(
        identity, _rational_multiply(long, long), right_scale=-1.0
    )
    source_to_bridge = _rational_scale(
        _rational_multiply(short, nut_comb), 0.5
    )
    transfer = _rational_multiply(source_to_bridge, closed_loop)
    leading = transfer.denominator[:, :1]
    if bool((leading.detach().abs() < 1e-14).any()):
        raise FloatingPointError("dynamic waveguide transfer is singular")
    return transfer.numerator / leading, transfer.denominator[:, 1:] / leading


def render_all_controls(unit: Tensor) -> Tensor:
    controls = decode_controls(unit)
    source = dynamic_excitation(controls)
    numerator, denominator = dynamic_transfer_rows(controls, WaveguideConfig())
    result = _df2_filter(source, numerator, denominator)
    if not bool(torch.isfinite(result.detach()).all()):
        raise FloatingPointError("all-control renderer produced non-finite audio")
    return result


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


def signature() -> tuple[str, dict[str, str]]:
    paths = [
        Path(__file__),
        Path(__file__).with_name("fixed_gradient_assessment.py"),
        *(Path(__file__).parents[1] / "src" / name for name in (
            "losses.py",
            "metrics.py",
            "synth.py",
            "exciter.py",
            "waveguide.py",
            "_fractional.py",
            "config.py",
            "runtime.py",
        )),
    ]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    library = Path(__file__).parents[1] / "external" / "cels" / "src" / "cels"
    hashes.update(
        {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(library.glob("*.py"))
        }
    )
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), hashes


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
    dynamic_audio = _df2_filter(
        dynamic_source, dynamic_numerator, dynamic_denominator
    )
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


def compute_persistent(root: Path, target_index: int, batch: int, device: str) -> None:
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
    assignments, ties = zip(
        *(hungarian_assignment(row, target) for row in positions), strict=True
    )
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
    collected = {"persistent_target": [], "all_controls": [], "timbre_only": []}
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
        collected["all_controls"].append(
            cosine(data["gradients"], data["candidates"], target)
        )
        collected["timbre_only"].append(
            cosine(
                data["gradients"][..., TIMBRE_INDICES],
                data["candidates"][:, TIMBRE_INDICES],
                target[TIMBRE_INDICES],
            )
        )
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
                    "positive_percent": float(100.0 * (selected > 0).mean()),
                    "median": float(np.median(selected)),
                    "eligible": len(selected),
                }
            )
    write_csv(output / "summary.csv", rows)
    by_key = {(row["loss"], row["condition"]): row for row in rows}
    labels = {
        "persistent_target": "Persistent target, N=2",
        "all_controls": "All seven controls, N=1",
        "timbre_only": "Timbre projection, N=1",
    }
    markdown = [
        "# Additional gradient diagnostics",
        "",
        "Mean phrase-wide cosine ± sample SD; the percentage of positive cosines is in parentheses.",
        "",
        "| Loss | " + " | ".join(labels.values()) + " |",
        "|---|" + "---:|" * len(labels),
    ]
    display = dict(zip(NAMES, (
        "L1", "L2", "SS", "MSS", "SmoMSS", "SOT", "SOT-NC", "TFW2",
        "logTFW2", "CeL", "logCeL", "decCeL", "tlogCeL",
    ), strict=True))
    for loss in NAMES:
        cells = []
        for condition in labels:
            row = by_key[(loss, condition)]
            cells.append(
                f"{row['mean']:.3f} ± {row['sample_sd']:.3f} "
                f"({row['positive_percent']:.1f}%)"
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
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
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
            compute_persistent(
                args.root, args.target_index, args.batch, args.device
            )
        else:
            compute_all_controls(
                args.root, args.target_index, args.batch, args.device
            )
    else:
        report(args.root, args.output)


if __name__ == "__main__":
    main()
