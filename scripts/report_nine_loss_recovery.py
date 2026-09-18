#!/usr/bin/env python3
"""Validate, aggregate, and render the fixed nine-loss recovery study."""

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
import torch
from icassp27_phrase.runtime import require_df2_backend
from icassp27_phrase.synth import PhraseSynth
from scipy.optimize import linear_sum_assignment

RAW = Path("results/nine-loss-recovery-packed/raw")
QUALIFICATION = Path("results/nine-loss-recovery-packed/qualification-cuda.json")
OUTPUT = Path("docs/nine-loss-recovery")
PAPER = Path("paper/figures")
METRICS = ("pitch_mae_cents", "onset_mae_ms", "log_spectral_distance_db")


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


def recompute_metrics(row: dict) -> tuple[float, float, list[int]]:
    f0 = np.asarray(row["reported_f0_hz"], dtype=np.float64)
    onset = np.asarray(row["reported_onset_seconds"], dtype=np.float64)
    target_f0 = np.asarray(row["target"]["f0_hz"], dtype=np.float64)
    target_onset = np.asarray(row["target"]["onset_seconds"], dtype=np.float64)
    pitch = np.log2(f0 / 80.0)
    target_pitch = np.log2(target_f0 / 80.0)
    cost = (pitch[:, None] - target_pitch[None]) ** 2 + (
        onset[:, None] - target_onset[None]
    ) ** 2
    _, assignment = linear_sum_assignment(cost)
    pitch_mae = float(np.abs(1200 * (pitch - target_pitch[assignment])).mean())
    onset_mae = float(np.abs(1000 * (onset - target_onset[assignment])).mean())
    return pitch_mae, onset_mae, assignment.tolist()


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
            payload["schema"] != "nine-loss-recovery-v2"
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
            pitch_mae, onset_mae, assignment = recompute_metrics(row)
            metrics = row["metrics"]
            if (
                assignment != metrics["assignment"]
                or abs(pitch_mae - metrics["pitch_mae_cents"]) > 1e-9
                or abs(onset_mae - metrics["onset_mae_ms"]) > 1e-9
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


def verify_common_targets(rows: dict[tuple[str, int, int], dict]) -> None:
    reference_loss = recovery.LOSSES[0]
    for cardinality in recovery.CARDINALITIES:
        for target in range(recovery.TARGETS_PER_CELL):
            reference = rows[reference_loss, cardinality, target]["target"]
            for loss in recovery.LOSSES[1:]:
                if rows[loss, cardinality, target]["target"] != reference:
                    raise ValueError(f"target differs across losses at C{cardinality}, T{target}")


def stft_magnitude(audio: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    return torch.stft(
        audio,
        n_fft=256,
        hop_length=64,
        win_length=256,
        window=window,
        center=True,
        return_complex=True,
    ).abs()


def compute_lsd(rows: dict[tuple[str, int, int], dict], device: str) -> dict:
    require_df2_backend(device)
    synth = PhraseSynth().to(device)
    window = torch.hann_window(256, periodic=True, dtype=torch.float64, device=device)
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
            target_magnitude = stft_magnitude(synth(target_f0, target_onset), window)
            floor = (target_magnitude.amax((-2, -1), keepdim=True) * 1e-5).clamp_min(
                torch.finfo(torch.float64).tiny
            )
            target_db = 20 * torch.log10(torch.maximum(target_magnitude, floor))
            for loss in recovery.LOSSES:
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
                candidate = stft_magnitude(synth(f0, onset), window)
                candidate_db = 20 * torch.log10(torch.maximum(candidate, floor))
                lsd = (candidate_db - target_db).square().mean((-2, -1)).sqrt().cpu()
                for index, value in enumerate(lsd.tolist()):
                    if not math.isfinite(value) or value < 0:
                        raise FloatingPointError(f"invalid LSD for {(loss, cardinality, index)}")
                    values[loss, cardinality, index] = value
    if set(values) != expected_keys():
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
            for loss, label in zip(recovery.LOSSES, recovery.LABELS, strict=True):
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
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def summarize(rows: dict, lsd: dict) -> tuple[list[dict], dict]:
    records = []
    medians = defaultdict(dict)
    for loss, label in zip(recovery.LOSSES, recovery.LABELS, strict=True):
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


def render_table(medians: dict, path: Path) -> None:
    labels = (
        "SS",
        "LinMSS",
        "SmoMSS",
        r"$\mathrm{TF}\mathcal{W}_2$",
        r"log-$\mathrm{TF}\mathcal{W}_2$",
        r"Ce$\mathcal L$ (Ours)",
        r"logCe$\mathcal L$ (Ours)",
        r"decCe$\mathcal L$ (Ours)",
        r"tlogCe$\mathcal L$ (Ours)",
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
        zip(recovery.LOSSES, labels, configurations, strict=True)
    ):
        cells = []
        for metric in ("pitch_mae_cents", "onset_mae_ms"):
            cells.extend(
                format_value(medians[metric][loss, cardinality])
                for cardinality in recovery.CARDINALITIES
            )
        lines.append(
            " & ".join((label, *configuration, *cells)) + r" \\"
        )
        if index == 4:
            lines.append(r"\addlinespace[1pt]")
    lines.extend((r"\bottomrule", r"\end{tabularx}", r"\endgroup"))
    atomic_text(path, "\n".join(lines) + "\n")


def save_figure(figure, path: Path) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    try:
        metadata = {"CreationDate": None, "ModDate": None} if path.suffix == ".pdf" else None
        figure.savefig(temporary, dpi=300, bbox_inches="tight", metadata=metadata)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def render_lsd(lsd: dict, output_stem: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch

    labels = (
        "SS",
        "LinMSS",
        "SmoMSS",
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
            recovery.LOSSES,
            [cmap(value) for value in np.linspace(0.08, 0.92, len(recovery.LOSSES))],
            strict=True,
        )
    )
    figure, axis = plt.subplots(figsize=(3.45, 2.58))
    offsets = np.linspace(-0.36, 0.36, len(recovery.LOSSES))
    for loss_index, loss in enumerate(recovery.LOSSES):
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
    axis.set_xticks(range(len(recovery.CARDINALITIES)), recovery.CARDINALITIES)
    axis.set_xlabel("Events", fontsize=8)
    axis.set_ylabel("LSD (dB)", fontsize=8)
    axis.tick_params(labelsize=7, length=2.0, pad=1.2)
    axis.grid(axis="y", color="0.88", linewidth=0.45)
    axis.legend(
        handles=[
            Patch(facecolor=colors[loss], edgecolor="black", label=label)
            for loss, label in zip(recovery.LOSSES, labels, strict=True)
        ],
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        frameon=False,
        fontsize=5.2,
        handlelength=0.9,
        columnspacing=0.7,
        borderaxespad=0.0,
    )
    for spine in axis.spines.values():
        spine.set_linewidth(0.6)
    figure.subplots_adjust(left=0.15, right=0.99, bottom=0.17, top=0.72)
    paths = {suffix: output_stem.with_suffix(f".{suffix}") for suffix in ("pdf", "png")}
    for path in paths.values():
        save_figure(figure, path)
    plt.close(figure)
    return {suffix: sha256(path) for suffix, path in paths.items()}


def write_markdown(medians: dict, path: Path) -> None:
    lines = [
        "# Nine-loss phrase recovery",
        "",
        "Medians across 150 target phrases; each phrase metric is the mean absolute error "
        "after octave-second Hungarian assignment.",
        "",
        "| Loss | "
        + " | ".join(f"{cardinality} event" for cardinality in recovery.CARDINALITIES)
        + " |",
        "|---|" + "---:|" * len(recovery.CARDINALITIES),
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
        for loss, label in zip(recovery.LOSSES, recovery.LABELS, strict=True):
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
    rows, validation = validate()
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
        "schema": "nine-loss-recovery-report-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "fit_count": len(rows),
        "loss_count": len(recovery.LOSSES),
        "target_phrases_per_condition": recovery.TARGETS_PER_CELL,
        "cardinalities": recovery.CARDINALITIES,
        "losses": recovery.LOSSES,
        "reported_iterate": "strict lowest-loss iterate",
        "aggregation": (
            "median across phrases of within-phrase Hungarian-matched mean absolute error"
        ),
        "lsd": (
            "RMS difference between centered periodic-Hann FFT-256/hop-64 log-magnitude "
            "spectra with a common target-relative -100 dB floor"
        ),
        "validation": validation,
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
                "fits": len(rows),
                "shards": len(validation["raw_shards"]),
                "aggregate_gpu_hours": validation["aggregate_gpu_hours"],
                "artifacts": provenance["artifacts"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
