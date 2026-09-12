#!/usr/bin/env python3
"""Render the paper's descriptive table and LSD figure from a signed milestone."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

LOSS_NAMES = (
    "waveform_l1",
    "waveform_mse",
    "smooth_mss",
    "sot_published_composite",
    "log_jtfot",
    "bidirectional_cumulative_energy",
)
LOSS_LABELS = (
    r"$\mathcal{L}_1$",
    r"$\mathcal{L}_2$",
    "MSS",
    "SOT",
    r"$\mathrm{TF}\mathcal{W}_2$",
    r"$\mathrm{BiCu}\mathcal{L}$",
)
TEX_LOSS_LABELS = (
    r"\lossLone{}",
    r"\lossLtwo{}",
    r"\lossMSS{}",
    r"\lossSOT{}",
    r"\lossTFW{}",
    r"\lossBiCum{}",
)
CARDINALITIES = (1, 2, 4, 6, 8)
METRICS = (
    ("pitch_mae_cents", r"\Delta f_0", "Cents"),
    ("onset_mae_ms", r"\Delta t", "ms"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _signed_payload_sha256(payload: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    canonical = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def read_signed_json(path: Path, *, status: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != status:
        raise ValueError(f"{path} does not have status={status!r}")
    observed = payload.get("payload_sha256")
    expected = _signed_payload_sha256(payload)
    if observed != expected:
        raise ValueError(f"{path} has an invalid payload signature")
    return payload


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


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    signed = dict(payload)
    signed["payload_sha256"] = _signed_payload_sha256(payload)
    atomic_text(path, json.dumps(signed, indent=2, sort_keys=True, allow_nan=False) + "\n")


def save_figure(figure: Any, path: Path) -> None:
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


def load_rows(
    per_phrase: Path,
    summary: dict[str, Any],
    health: dict[str, Any],
    milestone_n: int,
) -> list[dict[str, str]]:
    expected_fits = milestone_n * len(LOSS_NAMES) * len(CARDINALITIES)
    for name, payload, count_key in (
        ("report", summary, "raw_fit_count"),
        ("health", health, "fit_count"),
    ):
        if payload.get("milestone_n") != milestone_n:
            raise ValueError(f"{name} milestone does not equal n={milestone_n}")
        if payload.get("condition_count") != 30 or payload.get(count_key) != expected_fits:
            raise ValueError(f"{name} has the wrong condition or fit count")
        if payload.get("phrases_per_condition") != milestone_n:
            raise ValueError(f"{name} has the wrong phrases-per-condition count")
    expected_csv_hash = summary.get("artifacts", {}).get("per_phrase_csv_sha256")
    if sha256(per_phrase) != expected_csv_hash:
        raise ValueError("per_phrase.csv does not match the signed report")

    with per_phrase.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected_fits:
        raise ValueError(f"expected {expected_fits} per-phrase rows, found {len(rows)}")

    grouped_targets: dict[tuple[str, int], set[int]] = {
        (loss, cardinality): set()
        for loss in LOSS_NAMES
        for cardinality in CARDINALITIES
    }
    finite_fields = (
        "pitch_mae_cents",
        "onset_mae_ms",
        "log_spectral_distance_db",
    )
    for row in rows:
        key = (row["loss"], int(row["cardinality"]))
        if key not in grouped_targets:
            raise ValueError(f"unexpected condition {key}")
        target_index = int(row["target_index"])
        if target_index in grouped_targets[key]:
            raise ValueError(f"duplicate target index {target_index} in {key}")
        grouped_targets[key].add(target_index)
        if not all(math.isfinite(float(row[field])) for field in finite_fields):
            raise FloatingPointError(f"non-finite reported metric in {key}")
    expected_targets = set(range(milestone_n))
    if any(targets != expected_targets for targets in grouped_targets.values()):
        raise ValueError("one or more conditions do not contain exactly target indices 0..n-1")
    return rows


def format_table_value(value: float) -> str:
    rendered = f"{value:.3g}"
    if "e" in rendered:
        mantissa, exponent = rendered.split("e", 1)
        rendered = f"{mantissa}e{int(exponent)}"
    return rendered


def render_table(rows: list[dict[str, str]], output: Path) -> dict[str, Any]:
    grouped: dict[tuple[str, int, str], list[float]] = {
        (loss, cardinality, metric): []
        for loss in LOSS_NAMES
        for cardinality in CARDINALITIES
        for metric, _, _ in METRICS
    }
    for row in rows:
        loss = row["loss"]
        cardinality = int(row["cardinality"])
        for metric, _, _ in METRICS:
            grouped[(loss, cardinality, metric)].append(float(row[metric]))
    medians = {
        key: float(np.median(np.asarray(values, dtype=np.float64)))
        for key, values in grouped.items()
    }

    lines = [
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\renewcommand{\arraystretch}{1.02}",
        r"\begin{tabularx}{\columnwidth}{@{}l*{5}{>{\centering\arraybackslash}X}@{}}",
        r"\toprule",
        r"Loss & 1 Event & 2 Events & 4 Events & 6 Events & 8 Events \\",
        r"\midrule",
    ]
    ranking: dict[str, dict[str, dict[str, str]]] = {}
    serializable_medians: dict[str, dict[str, dict[str, float]]] = {}
    for metric_index, (metric, symbol, unit) in enumerate(METRICS):
        lines.append(rf"& \multicolumn{{5}}{{c}}{{${symbol}$ ({unit})}} \\")
        rank_by_cardinality: dict[int, dict[int, str]] = {}
        for cardinality in CARDINALITIES:
            values = np.asarray(
                [medians[(loss, cardinality, metric)] for loss in LOSS_NAMES],
                dtype=np.float64,
            )
            order = np.argsort(values, kind="stable")
            rank_by_cardinality[cardinality] = {
                int(order[0]): "best",
                int(order[1]): "second",
                int(order[2]): "third",
            }
        ranking[metric] = {
            str(cardinality): {
                LOSS_NAMES[index]: rank
                for index, rank in rank_by_cardinality[cardinality].items()
            }
            for cardinality in CARDINALITIES
        }
        serializable_medians[metric] = {
            loss: {
                str(cardinality): medians[(loss, cardinality, metric)]
                for cardinality in CARDINALITIES
            }
            for loss in LOSS_NAMES
        }
        for loss_index, (loss, label) in enumerate(
            zip(LOSS_NAMES, TEX_LOSS_LABELS, strict=True)
        ):
            cells: list[str] = []
            for cardinality in CARDINALITIES:
                value = format_table_value(medians[(loss, cardinality, metric)])
                rank = rank_by_cardinality[cardinality].get(loss_index)
                if rank == "best":
                    value = r"{\bfseries\boldmath $" + value + "$}"
                elif rank == "second":
                    value = r"\underline{" + value + "}"
                elif rank == "third":
                    value = r"\textit{" + value + "}"
                cells.append(value)
            lines.append(label + " & " + " & ".join(cells) + r" \\")
        if metric_index == 0:
            lines.extend((r"\addlinespace[2pt]", r"\midrule", r"\addlinespace[1pt]"))
    lines.extend((r"\bottomrule", r"\end{tabularx}", r"\endgroup"))
    atomic_text(output, "\n".join(lines) + "\n")
    return {
        "aggregation": (
            "median across phrases of within-phrase Hungarian-matched mean absolute error"
        ),
        "medians": serializable_medians,
        "ranking": ranking,
        "sha256": sha256(output),
    }


def render_lsd(rows: list[dict[str, str]], output_stem: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch

    groups: dict[tuple[str, int], list[float]] = {
        (loss, cardinality): []
        for loss in LOSS_NAMES
        for cardinality in CARDINALITIES
    }
    for row in rows:
        groups[(row["loss"], int(row["cardinality"]))].append(
            float(row["log_spectral_distance_db"])
        )

    cmap = plt.get_cmap("magma")
    colors = dict(
        zip(
            LOSS_NAMES,
            [cmap(value) for value in np.linspace(0.12, 0.88, 6)],
            strict=True,
        )
    )
    figure, axis = plt.subplots(figsize=(3.45, 2.42))
    offsets = np.linspace(-0.34, 0.34, len(LOSS_NAMES))
    for loss_index, loss_name in enumerate(LOSS_NAMES):
        for cardinality_index, cardinality in enumerate(CARDINALITIES):
            values = np.asarray(groups[(loss_name, cardinality)], dtype=np.float64)
            position = cardinality_index + offsets[loss_index]
            violin = axis.violinplot(
                values,
                positions=[position],
                widths=0.13,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in violin["bodies"]:
                body.set_facecolor(colors[loss_name])
                body.set_edgecolor("black")
                body.set_linewidth(0.4)
                body.set_alpha(0.62)
            lower, median, upper = np.quantile(values, (0.25, 0.5, 0.75))
            axis.vlines(position, lower, upper, color="white", linewidth=0.95)
            axis.scatter(
                position,
                median,
                s=10,
                color="white",
                edgecolor="black",
                linewidth=0.45,
                zorder=4,
            )
    axis.set_xticks(range(len(CARDINALITIES)), [str(value) for value in CARDINALITIES])
    axis.set_xlabel("Events", fontsize=8)
    axis.set_ylabel("LSD (dB)", fontsize=8)
    axis.tick_params(labelsize=7, length=2.0, pad=1.2)
    axis.grid(axis="y", color="0.88", linewidth=0.45)
    axis.legend(
        handles=[
            Patch(facecolor=colors[name], edgecolor="black", label=label)
            for name, label in zip(LOSS_NAMES, LOSS_LABELS, strict=True)
        ],
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        frameon=False,
        fontsize=6.6,
        handlelength=1.15,
        columnspacing=1.1,
        borderaxespad=0.0,
    )
    for spine in axis.spines.values():
        spine.set_linewidth(0.6)
    figure.subplots_adjust(left=0.15, right=0.99, bottom=0.17, top=0.79)
    paths = {suffix: output_stem.with_suffix(f".{suffix}") for suffix in ("pdf", "png")}
    for path in paths.values():
        save_figure(figure, path)
    plt.close(figure)
    return {suffix: sha256(path) for suffix, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-phrase", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--health", type=Path, required=True)
    parser.add_argument("--milestone", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.milestone < 1:
        raise ValueError("milestone must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = read_signed_json(args.summary, status="complete")
    health = read_signed_json(args.health, status="passed")
    rows = load_rows(args.per_phrase, summary, health, args.milestone)
    table_path = args.output_dir / "median_recovery_table.tex"
    table = render_table(rows, table_path)
    figure = render_lsd(rows, args.output_dir / "lsd_violin")
    provenance = {
        "schema": "paper-descriptive-milestone-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "milestone_n": args.milestone,
        "condition_count": 30,
        "fit_count": len(rows),
        "selection": "target_index < milestone_n within every cardinality",
        "sources": {
            "per_phrase_csv": str(args.per_phrase),
            "per_phrase_csv_sha256": sha256(args.per_phrase),
            "report": str(args.summary),
            "report_sha256": sha256(args.summary),
            "report_payload_sha256": summary["payload_sha256"],
            "health": str(args.health),
            "health_sha256": sha256(args.health),
            "health_payload_sha256": health["payload_sha256"],
        },
        "table": table,
        "figure4": {
            "metric": "complete per-phrase log-spectral-distance distributions",
            "artifacts": figure,
        },
    }
    atomic_json(args.output_dir / "descriptive_results.provenance.json", provenance)
    print(json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
