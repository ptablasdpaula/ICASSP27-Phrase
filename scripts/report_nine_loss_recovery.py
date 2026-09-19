#!/usr/bin/env python3
"""Validate the full recovery archive and report the screened objectives."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import run_nine_loss_recovery as recovery
import run_packed_nine_loss_recovery as packed
import run_sot_recovery_addon as sot_addon
import torch
from icassp27_phrase.config import MASTER_SEED
from icassp27_phrase.metrics import log_spectral_distance, recovery_metrics
from icassp27_phrase.runtime import require_df2_backend
from icassp27_phrase.synth import PhraseSynth

RAW = Path("results/phrase-recovery-16k/raw")
QUALIFICATION = Path("results/phrase-recovery-16k/qualification-cuda.json")
SOT_QUALIFICATION = Path("results/phrase-recovery-16k-sot/qualification.json")
OUTPUT = Path("docs/phrase-recovery/16k")
PAPER = Path("paper/figures")
METRICS = ("pitch_mae_cents", "onset_mae_ms", "log_spectral_distance_db")
REPORT_LOSSES = (
    "single_stft",
    "smooth_mss",
    "sot_published_composite",
    "linear_jtfot",
    "log_jtfot",
    "cel",
    "log_cel",
    "dec_cel",
    "tlog_cel",
)
REPORT_LABELS = (
    "SS",
    "SmoMSS",
    "SOT",
    "TFW2",
    "logTFW2",
    "CeL",
    "logCeL",
    "decCeL",
    "tlogCeL",
)
LOW_ALIGNMENT_CONTROLS = ("sot_published_composite", "linear_jtfot")
EXCLUDED_AFTER_SCREEN = ("mss",)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
        os.replace(temporary, path)
        os.chmod(path, 0o644)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def expected_keys() -> set[tuple[str, int, int]]:
    return {
        (loss, cardinality, target)
        for loss in recovery.LOSSES
        for cardinality in recovery.CARDINALITIES
        for target in range(recovery.TARGETS_PER_CELL)
    }


def reported_keys() -> set[tuple[str, int, int]]:
    return {
        (loss, cardinality, target)
        for loss in REPORT_LOSSES
        for cardinality in recovery.CARDINALITIES
        for target in range(recovery.TARGETS_PER_CELL)
    }


def random_lsd_keys() -> set[tuple[str, int, int]]:
    return {
        ("random", cardinality, target)
        for cardinality in recovery.CARDINALITIES
        for target in range(recovery.TARGETS_PER_CELL)
    }


def random_derangement(cardinality: int) -> np.ndarray:
    """Return a fixed no-self-match permutation for the random LSD baseline."""
    size = recovery.TARGETS_PER_CELL
    original = np.arange(size)
    generator = np.random.default_rng(MASTER_SEED + cardinality)
    while True:
        permutation = generator.permutation(size)
        if np.all(permutation != original):
            return permutation


def validate() -> tuple[dict[tuple[str, int, int], dict], dict]:
    paths = sorted(RAW.glob("*/*.json.gz"))
    if len(paths) != packed.TOTAL_SHARDS:
        raise ValueError(f"expected {packed.TOTAL_SHARDS} shards, found {len(paths)}")
    qualification = json.loads(QUALIFICATION.read_text())
    signature = qualification["scientific_signature"]
    execution_plan = packed.plan_hash()
    if qualification["execution_plan_sha256"] != execution_plan:
        raise ValueError("qualification belongs to a different packed execution plan")

    rows: dict[tuple[str, int, int], dict] = {}
    manifest = []
    source_commits: Counter[str] = Counter()
    gpus: Counter[str] = Counter()
    stop_reasons: Counter[str] = Counter()
    schedules: set[str] = set()
    total_wall_seconds = 0.0
    source_hashes = None
    for path in paths:
        with gzip.open(path, "rt") as stream:
            payload = json.load(stream)
        task = payload["packed_task"]
        if path != packed.expected_path(task):
            raise ValueError(f"task {task} is stored at an unexpected path")
        loss, cardinality, begin, count = packed.SHARD_SPECS[task]
        expected_range = [begin, begin + count]
        if (payload["loss"], payload["cardinality"], payload["target_range"]) != (
            loss,
            cardinality,
            expected_range,
        ):
            raise ValueError(f"task coordinates changed in {path}")
        if (
            payload["schema"] != "phrase-recovery-16k-v1"
            or payload["signature"] != signature
            or payload["execution_plan"] != "packed-nine-loss-recovery-v1"
            or payload["execution_plan_sha256"] != execution_plan
        ):
            raise ValueError(f"provenance mismatch in {path}")
        if len(payload["rows"]) != count:
            raise ValueError(f"row count mismatch in {path}")
        if source_hashes is None:
            source_hashes = payload["source_hashes"]
        elif payload["source_hashes"] != source_hashes:
            raise ValueError(f"scientific source hashes differ in {path}")
        source_commits[payload["source_commit"]] += 1
        gpus[payload["gpu"]] += 1
        schedules.add(json.dumps(payload["schedule"], sort_keys=True))
        total_wall_seconds += payload["wall_seconds"]
        manifest.append({"path": str(path), "sha256": sha256(path), "task": task})

        for offset, row in enumerate(payload["rows"]):
            target = begin + offset
            key = (loss, cardinality, target)
            if key in rows:
                raise ValueError(f"duplicate fit {key}")
            if (
                row["target_id"] != f"C{cardinality:02d}-T{target:04d}"
                or row["target"]["index"] != target + 1
                or row["target"]["cardinality"] != cardinality
            ):
                raise ValueError(f"target identity mismatch in {key}")
            scalars = [row[name] for name in ("initial_loss", "reported_loss", "terminal_loss")]
            if not all(math.isfinite(value) and value >= 0 for value in scalars):
                raise FloatingPointError(f"invalid objective value in {key}")
            if row["reported_loss"] > min(row["initial_loss"], row["terminal_loss"]) + 1e-12:
                raise ValueError(f"reported iterate is not the retained strict best in {key}")
            if not 0 < row["updates"] <= recovery.SCHEDULE.maximum_updates:
                raise ValueError(f"invalid update count in {key}")
            if row["stopped_by"] not in {"patience", "maximum_updates"}:
                raise ValueError(f"invalid stopping reason in {key}")
            recomputed = recovery_metrics(
                np.asarray(row["reported_f0_hz"]),
                np.asarray(row["reported_onset_seconds"]),
                np.asarray(row["target"]["f0_hz"]),
                np.asarray(row["target"]["onset_seconds"]),
            )
            metrics = row["metrics"]
            if (
                recomputed["assignment"] != metrics["assignment"]
                or recomputed["assignment_tied"] != metrics["assignment_tied"]
                or abs(recomputed["pitch_mae_cents"] - metrics["pitch_mae_cents"]) > 1e-9
                or abs(recomputed["onset_mae_ms"] - metrics["onset_mae_ms"]) > 1e-9
            ):
                raise ValueError(f"stored Hungarian metrics do not reproduce in {key}")
            stop_reasons[row["stopped_by"]] += 1
            rows[key] = row

    if set(rows) != expected_keys():
        raise ValueError("packed outputs do not cover every expected fit exactly once")
    if len(schedules) != 1:
        raise ValueError("fits used more than one optimisation schedule")
    metadata = {
        "scientific_signature": signature,
        "execution_plan_sha256": execution_plan,
        "source_hashes": source_hashes,
        "source_commits": dict(source_commits),
        "gpus": dict(gpus),
        "stop_reasons": dict(stop_reasons),
        "aggregate_gpu_hours": total_wall_seconds / 3600,
        "raw_shards": manifest,
    }
    return rows, metadata


def validate_sot() -> tuple[dict[tuple[str, int, int], dict], dict]:
    paths = sorted((sot_addon.ROOT / "raw").glob("*/*.json.gz"))
    if len(paths) != sot_addon.TOTAL_SHARDS:
        raise ValueError(f"expected {sot_addon.TOTAL_SHARDS} SOT shards, found {len(paths)}")
    qualification = json.loads(SOT_QUALIFICATION.read_text())
    signature = qualification["scientific_signature"]
    execution_plan = sot_addon.plan_hash()
    if not qualification["passed"] or qualification["execution_plan_sha256"] != execution_plan:
        raise ValueError("SOT qualification is missing or belongs to another execution plan")

    rows: dict[tuple[str, int, int], dict] = {}
    manifest = []
    source_commits: Counter[str] = Counter()
    gpus: Counter[str] = Counter()
    stop_reasons: Counter[str] = Counter()
    schedules: set[str] = set()
    total_wall_seconds = 0.0
    for path in paths:
        with gzip.open(path, "rt") as stream:
            payload = json.load(stream)
        task = payload["addon_task"]
        if path != sot_addon.expected_path(task):
            raise ValueError(f"SOT task {task} is stored at an unexpected path")
        loss, cardinality, begin, count = sot_addon.SHARD_SPECS[task]
        if (payload["loss"], payload["cardinality"], payload["target_range"]) != (
            loss,
            cardinality,
            [begin, begin + count],
        ):
            raise ValueError(f"SOT task coordinates changed in {path}")
        if (
            payload["schema"] != "phrase-recovery-16k-sot-addon-v1"
            or payload["signature"] != signature
            or payload["source_hashes"] != qualification["source_hashes"]
            or payload["execution_plan"] != "sot-recovery-addon-v1"
            or payload["execution_plan_sha256"] != execution_plan
            or payload["addon_source_sha256"] != execution_plan
            or len(payload["rows"]) != count
        ):
            raise ValueError(f"SOT provenance mismatch in {path}")
        source_commits[payload["source_commit"]] += 1
        gpus[payload["gpu"]] += 1
        schedules.add(json.dumps(payload["schedule"], sort_keys=True))
        total_wall_seconds += payload["wall_seconds"]
        manifest.append({"path": str(path), "sha256": sha256(path), "task": task})
        for offset, row in enumerate(payload["rows"]):
            target = begin + offset
            key = (loss, cardinality, target)
            if key in rows:
                raise ValueError(f"duplicate SOT fit {key}")
            recomputed = recovery_metrics(
                np.asarray(row["reported_f0_hz"]),
                np.asarray(row["reported_onset_seconds"]),
                np.asarray(row["target"]["f0_hz"]),
                np.asarray(row["target"]["onset_seconds"]),
            )
            metrics = row["metrics"]
            if (
                row["target_id"] != f"C{cardinality:02d}-T{target:04d}"
                or row["target"]["index"] != target + 1
                or row["target"]["cardinality"] != cardinality
                or recomputed["assignment"] != metrics["assignment"]
                or recomputed["assignment_tied"] != metrics["assignment_tied"]
                or abs(recomputed["pitch_mae_cents"] - metrics["pitch_mae_cents"]) > 1e-9
                or abs(recomputed["onset_mae_ms"] - metrics["onset_mae_ms"]) > 1e-9
            ):
                raise ValueError(f"SOT row does not reproduce in {key}")
            scalars = [row[name] for name in ("initial_loss", "reported_loss", "terminal_loss")]
            if not all(math.isfinite(value) and value >= 0 for value in scalars):
                raise FloatingPointError(f"invalid SOT objective value in {key}")
            if row["reported_loss"] > min(row["initial_loss"], row["terminal_loss"]) + 1e-12:
                raise ValueError(f"reported SOT iterate is not the retained strict best in {key}")
            if not 0 < row["updates"] <= recovery.SCHEDULE.maximum_updates:
                raise ValueError(f"invalid SOT update count in {key}")
            if row["stopped_by"] not in {"patience", "maximum_updates"}:
                raise ValueError(f"invalid SOT stopping reason in {key}")
            stop_reasons[row["stopped_by"]] += 1
            rows[key] = row

    expected = {
        (sot_addon.LOSS, cardinality, target)
        for cardinality in recovery.CARDINALITIES
        for target in range(recovery.TARGETS_PER_CELL)
    }
    if set(rows) != expected or len(schedules) != 1:
        raise ValueError("SOT outputs do not cover every expected fit under one schedule")
    return rows, {
        "scientific_signature": signature,
        "execution_plan_sha256": execution_plan,
        "source_commits": dict(source_commits),
        "gpus": dict(gpus),
        "stop_reasons": dict(stop_reasons),
        "aggregate_gpu_hours": total_wall_seconds / 3600,
        "raw_shards": manifest,
    }


def verify_common_targets(rows: dict[tuple[str, int, int], dict]) -> None:
    reference_loss = recovery.LOSSES[0]
    compared_losses = (*recovery.LOSSES[1:], sot_addon.LOSS)
    for cardinality in recovery.CARDINALITIES:
        for target in range(recovery.TARGETS_PER_CELL):
            reference = rows[reference_loss, cardinality, target]["target"]
            for loss in compared_losses:
                if rows[loss, cardinality, target]["target"] != reference:
                    raise ValueError(f"target differs across losses at C{cardinality}, T{target}")


def compute_lsd(rows: dict[tuple[str, int, int], dict], device: str) -> dict:
    require_df2_backend(device)
    synth = PhraseSynth().to(device)
    values = {}
    with torch.no_grad():
        for cardinality in recovery.CARDINALITIES:
            target_rows = [
                rows[recovery.LOSSES[0], cardinality, index]
                for index in range(recovery.TARGETS_PER_CELL)
            ]
            target_f0 = torch.tensor(
                [row["target"]["f0_hz"] for row in target_rows],
                dtype=torch.float64,
                device=device,
            )
            target_onset = torch.tensor(
                [row["target"]["onset_seconds"] for row in target_rows],
                dtype=torch.float64,
                device=device,
            )
            target_audio = synth(target_f0, target_onset)
            permutation = torch.as_tensor(
                random_derangement(cardinality), dtype=torch.long, device=device
            )
            random_lsd = log_spectral_distance(target_audio[permutation], target_audio).cpu()
            for index, value in enumerate(random_lsd.tolist()):
                if not math.isfinite(value) or value < 0:
                    raise FloatingPointError(
                        f"invalid random LSD for {(cardinality, index)}"
                    )
                values["random", cardinality, index] = value
            for loss in REPORT_LOSSES:
                group = [
                    rows[loss, cardinality, index]
                    for index in range(recovery.TARGETS_PER_CELL)
                ]
                f0 = torch.tensor(
                    [row["reported_f0_hz"] for row in group],
                    dtype=torch.float64,
                    device=device,
                )
                onset = torch.tensor(
                    [row["reported_onset_seconds"] for row in group],
                    dtype=torch.float64,
                    device=device,
                )
                candidate_audio = synth(f0, onset)
                lsd = log_spectral_distance(candidate_audio, target_audio).cpu()
                for index, value in enumerate(lsd.tolist()):
                    if not math.isfinite(value) or value < 0:
                        raise FloatingPointError(f"invalid LSD for {(loss, cardinality, index)}")
                    values[loss, cardinality, index] = value
    if set(values) != reported_keys() | random_lsd_keys():
        raise ValueError("LSD pass did not cover every fit")
    return values


def write_per_phrase(rows: dict, lsd: dict, path: Path) -> None:
    fields = (
        "loss",
        "label",
        "cardinality",
        "target_index",
        "target_id",
        "pitch_mae_cents",
        "onset_mae_ms",
        "log_spectral_distance_db",
        "initial_loss",
        "reported_loss",
        "terminal_loss",
        "updates",
        "stopped_by",
        "lr_reductions",
        "target_f0_hz",
        "target_onset_seconds",
        "reported_f0_hz",
        "reported_onset_seconds",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for loss, label in zip(REPORT_LOSSES, REPORT_LABELS, strict=True):
                for cardinality in recovery.CARDINALITIES:
                    for target in range(recovery.TARGETS_PER_CELL):
                        row = rows[loss, cardinality, target]
                        writer.writerow(
                            {
                                "loss": loss,
                                "label": label,
                                "cardinality": cardinality,
                                "target_index": target,
                                "target_id": row["target_id"],
                                "pitch_mae_cents": row["metrics"]["pitch_mae_cents"],
                                "onset_mae_ms": row["metrics"]["onset_mae_ms"],
                                "log_spectral_distance_db": lsd[loss, cardinality, target],
                                "initial_loss": row["initial_loss"],
                                "reported_loss": row["reported_loss"],
                                "terminal_loss": row["terminal_loss"],
                                "updates": row["updates"],
                                "stopped_by": row["stopped_by"],
                                "lr_reductions": row["lr_reductions"],
                                "target_f0_hz": json.dumps(row["target"]["f0_hz"]),
                                "target_onset_seconds": json.dumps(
                                    row["target"]["onset_seconds"]
                                ),
                                "reported_f0_hz": json.dumps(row["reported_f0_hz"]),
                                "reported_onset_seconds": json.dumps(
                                    row["reported_onset_seconds"]
                                ),
                            }
                        )
        os.replace(temporary, path)
        os.chmod(path, 0o644)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def summarize(rows: dict, lsd: dict) -> tuple[list[dict], dict]:
    records = []
    medians = defaultdict(dict)
    for loss, label in zip(REPORT_LOSSES, REPORT_LABELS, strict=True):
        for cardinality in recovery.CARDINALITIES:
            for metric in METRICS:
                if metric == "log_spectral_distance_db":
                    values = np.asarray(
                        [lsd[loss, cardinality, index] for index in range(150)]
                    )
                else:
                    values = np.asarray(
                        [
                            rows[loss, cardinality, index]["metrics"][metric]
                            for index in range(150)
                        ]
                    )
                record = {
                    "loss": loss,
                    "label": label,
                    "cardinality": cardinality,
                    "metric": metric,
                    "n": len(values),
                    "mean": float(values.mean()),
                    "sample_std": float(values.std(ddof=1)),
                    "q1": float(np.quantile(values, 0.25)),
                    "median": float(np.median(values)),
                    "q3": float(np.quantile(values, 0.75)),
                }
                records.append(record)
                medians[metric][loss, cardinality] = record["median"]
    return records, medians


def write_summary(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
        os.replace(temporary, path)
        os.chmod(path, 0o644)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def format_value(value: float) -> str:
    if value < 0.01:
        return r"$<$.01"
    rendered = f"{value:.3g}"
    if "e" in rendered:
        mantissa, exponent = rendered.split("e", 1)
        rendered = f"{mantissa}e{int(exponent)}"
    return rendered


def format_ranked_value(value: float, rank: int) -> str:
    rendered = format_value(value)
    if rank == 1:
        return rf"\textbf{{{rendered}}}"
    if rank == 2:
        return rf"\underline{{{rendered}}}"
    if rank == 3:
        return rf"\textit{{{rendered}}}"
    return rendered


def render_table(medians: dict, path: Path) -> None:
    labels = (
        "SS",
        "SmoMSS",
        "SOT",
        r"$\mathrm{TF}\mathcal{W}_2$",
        r"log-$\mathrm{TF}\mathcal{W}_2$",
        r"Ce$\mathcal L$",
        r"logCe$\mathcal L$",
        r"decCe$\mathcal L$",
        r"tlogCe$\mathcal L$",
    )
    configurations = (
        ("--", "--", "--"),
        ("--", "--", "--"),
        ("--", "--", "--"),
        ("--", "--", "--"),
        ("--", "--", "--"),
        (r"$\fourdiagonals$", "Lin.", r"$\times$"),
        (r"$\fourdiagonals$", "Log", r"$\times$"),
        (r"$\fourdiagonals$", "Lin.", r"$\checkmark$"),
        (r"$\forwarddiagonals$", "Log", r"$\times$"),
    )
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1.25pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\newcommand{\fourdiagonals}{\mathord{\ooalign{%",
        r"  \hfil$\nearrow$\hfil\cr",
        r"  \hfil$\nwarrow$\hfil\cr",
        r"  \hfil$\searrow$\hfil\cr",
        r"  \hfil$\swarrow$\hfil\cr}}}",
        r"\newcommand{\forwarddiagonals}{\mathord{\ooalign{%",
        r"  \hfil$\nearrow$\hfil\cr",
        r"  \hfil$\searrow$\hfil\cr}}}",
        (r"\begin{tabularx}{\linewidth}{@{}lccc*{5}{>{\centering\arraybackslash}X}|"
         r"*{5}{>{\centering\arraybackslash}X}@{}}"),
        r"& & &",
        (r"& \multicolumn{5}{c|}{$\Delta f_0$ (cents)} & "
         r"\multicolumn{5}{c}{$\Delta t$ (ms)} \\"),
        r"Loss & Directions & Weighing & Decay",
        (r"& 1 event & 2 events & 4 events & 6 events & 8 events "
         r"& 1 event & 2 events & 4 events & 6 events & 8 events \\"),
        r"\midrule",
    ]
    for index, (loss, label, configuration) in enumerate(
        zip(REPORT_LOSSES, labels, configurations, strict=True)
    ):
        cells = []
        for metric in ("pitch_mae_cents", "onset_mae_ms"):
            for cardinality in recovery.CARDINALITIES:
                values = {
                    candidate: medians[metric][candidate, cardinality]
                    for candidate in REPORT_LOSSES
                }
                # Displayed values below .01 share the best rank. Other exact
                # ties also share a rank, so the next distinct value follows.
                groups = sorted({0.0 if value < 0.01 else value for value in values.values()})
                value = values[loss]
                group = 0.0 if value < 0.01 else value
                cells.append(format_ranked_value(value, groups.index(group) + 1))
        lines.append(
            " & ".join((label, *configuration, *cells)) + r" \\"
        )
        if index == 4:
            lines.append(r"\midrule")
    lines.extend((r"\bottomrule", r"\end{tabularx}", r"\endgroup"))
    atomic_text(path, "\n".join(lines) + "\n")


def save_figure(figure, path: Path, *, pad_inches: float = 0.1) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    try:
        metadata = {"CreationDate": None, "ModDate": None} if path.suffix == ".pdf" else None
        figure.savefig(
            temporary,
            dpi=300,
            bbox_inches="tight",
            pad_inches=pad_inches,
            metadata=metadata,
        )
        os.replace(temporary, path)
        os.chmod(path, 0o644)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def render_lsd(lsd: dict, output_stem: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    labels = (
        "SS",
        "SmoMSS",
        "SOT",
        r"$\mathrm{TF}\mathcal{W}_2$",
        r"log-$\mathrm{TF}\mathcal{W}_2$",
        r"Ce$\mathcal{L}$",
        r"logCe$\mathcal{L}$",
        r"decCe$\mathcal{L}$",
        r"tlogCe$\mathcal{L}$",
    )
    cmap = plt.get_cmap("magma")
    colors = dict(
        zip(
            REPORT_LOSSES,
            [cmap(value) for value in np.linspace(0.08, 0.92, len(REPORT_LOSSES))],
            strict=True,
        )
    )
    figure, axis = plt.subplots(figsize=(3.5, 2.45))
    offsets = np.linspace(-0.36, 0.36, len(REPORT_LOSSES))
    for loss_index, loss in enumerate(REPORT_LOSSES):
        for cardinality_index, cardinality in enumerate(recovery.CARDINALITIES):
            values = np.asarray([lsd[loss, cardinality, index] for index in range(150)])
            position = cardinality_index + offsets[loss_index]
            violin = axis.violinplot(
                values,
                positions=[position],
                widths=0.082,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in violin["bodies"]:
                body.set_facecolor(colors[loss])
                body.set_edgecolor("black")
                body.set_linewidth(0.35)
                body.set_alpha(0.68)
            lower, median, upper = np.quantile(values, (0.25, 0.5, 0.75))
            axis.vlines(position, lower, upper, color="white", linewidth=0.85)
            axis.scatter(
                position,
                median,
                s=8,
                color="white",
                edgecolor="black",
                linewidth=0.35,
                zorder=4,
            )
    random_means = [
        np.mean(
            [
                lsd["random", cardinality, index]
                for index in range(recovery.TARGETS_PER_CELL)
            ]
        )
        for cardinality in recovery.CARDINALITIES
    ]
    axis.plot(
        range(len(recovery.CARDINALITIES)),
        random_means,
        color="0.2",
        linestyle="--",
        linewidth=0.9,
        zorder=5,
    )
    axis.set_xticks(range(len(recovery.CARDINALITIES)), recovery.CARDINALITIES)
    axis.set_xlabel("Number of events", fontsize=8, labelpad=5)
    axis.set_ylabel("LSD (dB)", fontsize=8)
    axis.tick_params(labelsize=7, length=2.0, pad=1.2)
    axis.grid(axis="y", color="0.88", linewidth=0.45)
    handles = [
        Patch(facecolor=colors[loss], edgecolor="black", label=label)
        for loss, label in zip(REPORT_LOSSES, labels, strict=True)
    ]
    handles.append(Line2D([0], [0], color="0.2", linestyle="--", label="Random"))
    axis.legend(
        handles=handles,
        ncol=len(handles),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
        fontsize=4.8,
        handlelength=0.45,
        handletextpad=0.18,
        columnspacing=0.25,
        borderaxespad=0.0,
    )
    for spine in axis.spines.values():
        spine.set_linewidth(0.6)
    figure.subplots_adjust(left=0.15, right=0.99, bottom=0.21, top=0.84)
    paths = {suffix: output_stem.with_suffix(f".{suffix}") for suffix in ("pdf", "png")}
    for path in paths.values():
        save_figure(figure, path, pad_inches=0.08)
    plt.close(figure)
    return {suffix: sha256(path) for suffix, path in paths.items()}


def write_markdown(medians: dict, path: Path) -> None:
    lines = [
        "# Screened phrase recovery",
        "",
        "Medians across 150 target phrases; each phrase metric is the mean absolute error "
        "after octave-second Hungarian assignment.",
    ]
    for metric, unit in (("pitch_mae_cents", "cents"), ("onset_mae_ms", "ms")):
        lines.extend(("", f"## {metric} ({unit})", ""))
        lines.extend(
            (
                "| Loss | "
                + " | ".join(str(cardinality) for cardinality in recovery.CARDINALITIES)
                + " |",
                "|---|" + "---:|" * len(recovery.CARDINALITIES),
            )
        )
        for loss, label in zip(REPORT_LOSSES, REPORT_LABELS, strict=True):
            values = " | ".join(
                f"{medians[metric][loss, cardinality]:.6g}"
                for cardinality in recovery.CARDINALITIES
            )
            lines.append(f"| {label} | {values} |")
    atomic_text(path, "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows, main_validation = validate()
    sot_rows, sot_validation = validate_sot()
    if set(rows) & set(sot_rows):
        raise ValueError("SOT add-on duplicates a main recovery key")
    rows.update(sot_rows)
    verify_common_targets(rows)
    lsd = compute_lsd(rows, args.device)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    per_phrase_path = OUTPUT / "per_phrase.csv"
    summary_path = OUTPUT / "summary.csv"
    table_path = PAPER / "median_recovery_table_pending.tex"
    write_per_phrase(rows, lsd, per_phrase_path)
    summary, medians = summarize(rows, lsd)
    write_summary(summary, summary_path)
    render_table(medians, table_path)
    figure = render_lsd(lsd, PAPER / "lsd_violin")
    write_markdown(medians, OUTPUT / "README.md")
    provenance = {
        "schema": "phrase-recovery-16k-report-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "fit_count": len(reported_keys()),
        "validated_fit_count": len(rows),
        "loss_count": len(REPORT_LOSSES),
        "target_phrases_per_condition": recovery.TARGETS_PER_CELL,
        "cardinalities": recovery.CARDINALITIES,
        "losses": REPORT_LOSSES,
        "low_alignment_controls": LOW_ALIGNMENT_CONTROLS,
        "excluded_after_single_event_screen": EXCLUDED_AFTER_SCREEN,
        "reported_iterate": "strict lowest-loss iterate",
        "aggregation": (
            "median across phrases of within-phrase Hungarian-matched mean absolute error"
        ),
        "lsd": (
            "RMS difference between centered periodic-Hann FFT-1024/hop-256 log-magnitude "
            "spectra with a common target-relative -100 dB floor"
        ),
        "random_lsd_baseline": (
            "mean LSD after pairing each target with a fixed, no-self-match random "
            "permutation of the other targets at the same event cardinality"
        ),
        "validation": {"main": main_validation, "sot_addon": sot_validation},
        "artifacts": {
            "per_phrase_csv": sha256(per_phrase_path),
            "summary_csv": sha256(summary_path),
            "paper_table": sha256(table_path),
            "lsd_figure": figure,
        },
    }
    atomic_json(OUTPUT / "provenance.json", provenance)
    print(
        json.dumps(
            {
                "status": "complete",
                "fits": len(reported_keys()),
                "validated_fits": len(rows),
                "shards": len(main_validation["raw_shards"])
                + len(sot_validation["raw_shards"]),
                "aggregate_gpu_hours": main_validation["aggregate_gpu_hours"]
                + sot_validation["aggregate_gpu_hours"],
                "artifacts": provenance["artifacts"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
