#!/usr/bin/env python3
"""Render the paper's eight-loss n=150 table and LSD figure."""

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
    "linear_jtfot",
    "log_jtfot",
    "bidirectional_cumulative_energy",
    "log_quadrature_bicul",
)
LOSS_LABELS = (
    r"$L_1$",
    r"$L_2$",
    "MSS",
    "SOT",
    r"$\mathrm{TF}\mathcal{W}_2$",
    r"$\hookrightarrow$ $1\,\mathrm{s}=1\,\mathrm{oct}$",
    r"$\mathrm{BiCu}\mathcal{L}$",
    r"$\hookrightarrow$ Log-Q",
)
TEX_LOSS_LABELS = (
    r"$L_1$",
    r"$L_2$",
    r"\acrshort{mss}",
    r"\acrshort{sot}",
    r"\acrshort{tfw}",
    r"$\hookrightarrow$ $1\,\mathrm{s}=1\,\mathrm{oct}$",
    r"\acrshort{bicul}",
    r"$\hookrightarrow$ \acrshort{logq}",
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
    milestone_n: int,
) -> list[dict[str, str]]:
    expected_fits = milestone_n * len(LOSS_NAMES) * len(CARDINALITIES)
    if (
        milestone_n != 150
        or summary.get("schema") != "eight-loss-confirmatory-report-v1"
        or summary.get("phrases_per_condition") != milestone_n
        or summary.get("condition_count") != 40
        or summary.get("loss_count") != 8
        or summary.get("fit_count") != expected_fits
        or tuple(summary.get("cardinalities", ())) != CARDINALITIES
        or tuple(summary.get("losses", ())) != LOSS_NAMES
        or summary.get("descriptive_only") is not True
    ):
        raise ValueError("report is not the complete eight-loss n=150 artifact")
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
        if row.get("target_id") != f"C{key[1]:02d}-T{target_index:04d}":
            raise ValueError(f"unexpected target identity in {key}")
        if target_index in grouped_targets[key]:
            raise ValueError(f"duplicate target index {target_index} in {key}")
        grouped_targets[key].add(target_index)
        values = [float(row[field]) for field in finite_fields]
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
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

    ranking: dict[str, dict[str, dict[str, str]]] = {}
    serializable_medians: dict[str, dict[str, dict[str, float]]] = {}
    ranks_by_metric: dict[str, dict[int, dict[int, str]]] = {}
    for metric, _, _ in METRICS:
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
        ranks_by_metric[metric] = rank_by_cardinality
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

    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.0pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        (r"\begin{tabularx}{\linewidth}"
         r"{@{}l*{5}{>{\centering\arraybackslash}X}|"
         r"*{5}{>{\centering\arraybackslash}X}@{}}"),
        r"\toprule",
        (r"Loss & \multicolumn{5}{c|}{$\Delta f_0$ (Cents)} & "
         r"\multicolumn{5}{c}{$\Delta t$ (ms)} \\"),
        (r"& 1 Event & 2 Events & 4 Events & 6 Events & 8 Events "
         r"& 1 Event & 2 Events & 4 Events & 6 Events & 8 Events \\"),
        r"\midrule",
    ]
    for loss_index, (loss, label) in enumerate(
        zip(LOSS_NAMES, TEX_LOSS_LABELS, strict=True)
    ):
        cells: list[str] = []
        for metric, _, _ in METRICS:
            for cardinality in CARDINALITIES:
                value = format_table_value(medians[(loss, cardinality, metric)])
                rank = ranks_by_metric[metric][cardinality].get(loss_index)
                if rank == "best":
                    value = r"\ensuremath{\mathbf{" + value + "}}"
                elif rank == "second":
                    value = r"\mbox{\underline{" + value + "}}"
                elif rank == "third":
                    value = r"\ensuremath{\mathit{" + value + "}}"
                else:
                    value = r"\mbox{" + value + "}"
                cells.append(value)
        lines.append(label + " & " + " & ".join(cells) + r" \\")
        if loss == "log_jtfot":
            lines.append(r"\addlinespace[1pt]")
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
            [cmap(value) for value in np.linspace(0.10, 0.90, len(LOSS_NAMES))],
            strict=True,
        )
    )
    figure, axis = plt.subplots(figsize=(3.45, 2.58))
    offsets = np.linspace(-0.36, 0.36, len(LOSS_NAMES))
    for loss_index, loss_name in enumerate(LOSS_NAMES):
        for cardinality_index, cardinality in enumerate(CARDINALITIES):
            values = np.asarray(groups[(loss_name, cardinality)], dtype=np.float64)
            position = cardinality_index + offsets[loss_index]
            violin = axis.violinplot(
                values,
                positions=[position],
                widths=0.090,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in violin["bodies"]:
                body.set_facecolor(colors[loss_name])
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
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        frameon=False,
        fontsize=5.5,
        handlelength=1.0,
        columnspacing=0.75,
        borderaxespad=0.0,
    )
    for spine in axis.spines.values():
        spine.set_linewidth(0.6)
    figure.subplots_adjust(left=0.15, right=0.99, bottom=0.17, top=0.77)
    paths = {suffix: output_stem.with_suffix(f".{suffix}") for suffix in ("pdf", "png")}
    for path in paths.values():
        save_figure(figure, path)
    plt.close(figure)
    return {suffix: sha256(path) for suffix, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-phrase", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--milestone", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.milestone != 150:
        raise ValueError("the combined eight-loss report is fixed at n=150")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = read_signed_json(args.summary, status="complete")
    rows = load_rows(args.per_phrase, summary, args.milestone)
    table_path = args.output_dir / "median_recovery_table.tex"
    table = render_table(rows, table_path)
    figure = render_lsd(rows, args.output_dir / "lsd_violin")
    provenance = {
        "schema": "paper-eight-loss-n150-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "milestone_n": args.milestone,
        "condition_count": 40,
        "loss_count": 8,
        "fit_count": len(rows),
        "selection": "target_index < milestone_n within every cardinality",
        "sources": {
            "per_phrase_csv": str(args.per_phrase),
            "per_phrase_csv_sha256": sha256(args.per_phrase),
            "report": str(args.summary),
            "report_sha256": sha256(args.summary),
            "report_payload_sha256": summary["payload_sha256"],
            "input_reports": summary["inputs"],
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
