"""Render saved phrase cosine ± SD with explicit colour anchors."""

import csv
import json
from pathlib import Path

import numpy as np
from extend_gradient_cardinality import EXTENDED_COLUMNS
from icassp27_phrase.gradient_assessment import NAMES
from report_gradient_alignment import LABELS, TEXT_LABELS, compact_cosine, render

EXTRA_NAME = "forward_log"
EXTRA_TEXT_LABEL = "tlogCeL"
EXTRA_LABEL = r"tlogCe$\mathcal{L}$"


def main():
    output = Path("docs/gradient-assessment/alignment-extended")
    with (output / "summary.csv").open() as file:
        base_records = [
            row for row in csv.DictReader(file) if row["metric"] == "phrase-cosine"
        ]
        rows = {
            (int(r["events"]), r["condition"], r["loss"]): r
            for r in base_records
        }
    extra_output = Path("docs/gradient-assessment/log-decay-directions")
    with (extra_output / "summary.csv").open() as file:
        extra_rows = {
            (int(r["events"]), r["condition"], r["loss"]): r for r in csv.DictReader(file)
        }
    names = (*NAMES, EXTRA_NAME)
    text_labels = (*TEXT_LABELS, EXTRA_TEXT_LABEL)
    labels = (*LABELS, EXTRA_LABEL)
    means = np.zeros((len(names), len(EXTENDED_COLUMNS)))
    sds = np.zeros_like(means)
    records = list(base_records)
    for i, loss in enumerate(NAMES):
        for j, (events, condition) in enumerate(EXTENDED_COLUMNS):
            row = rows[events, condition, loss]
            means[i, j], sds[i, j] = float(row["mean"]), float(row["candidate_sd"])
    for j, (events, condition) in enumerate(EXTENDED_COLUMNS):
        row = extra_rows[events, condition, EXTRA_NAME]
        means[-1, j] = float(row["mean_phrase_cosine"])
        sds[-1, j] = float(row["candidate_sd"])
        records.append(
            {
                "metric": "phrase-cosine",
                "events": events,
                "condition": condition,
                "loss": EXTRA_NAME,
                "mean": row["mean_phrase_cosine"],
                "candidate_sd": row["candidate_sd"],
                "eligible": row["eligible"],
                "targets": row["targets"],
                "candidates_per_target": row["candidates_per_target"],
            }
        )
    assert np.isfinite(means).all() and np.isfinite(sds).all()
    render(
        output,
        "phrase-cosine",
        means,
        sds,
        EXTENDED_COLUMNS,
        cmap_name="RdBu",
        cropped_colorbar=True,
        compact_layout=True,
        labels=labels,
    )
    headers = [
        f"{ {'joint': 'Joint', 'pitch': 'Known time', 'time': 'Known f0'}[c] }: {n}"
        for n, c in EXTENDED_COLUMNS
    ]
    lines = [
        "# Mean whole-phrase cosine ± SD",
        "",
        "| Loss | " + " | ".join(headers) + " |",
        "|---|" + "---:|" * len(EXTENDED_COLUMNS),
    ]
    for i, label in enumerate(text_labels):
        cells = [
            f"{compact_cosine(means[i, j])} ± {compact_cosine(sds[i, j])}"
            for j in range(len(EXTENDED_COLUMNS))
        ]
        lines.append("| " + " | ".join([label, *cells]) + " |")
    (output / "phrase-cosine.md").write_text("\n".join(lines) + "\n")
    with (output / "phrase-cosine.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output / "phrase-cosine-style.json").write_text(
        json.dumps(
            dict(
                cmap="RdBu",
                vmin=-1.0,
                observed_min=float(means.min()),
                vmax=1.0,
                colorbar_xlim=[-0.1, 1.0],
                ticks=[0, 0.5, 1],
                observed_max=float(means.max()),
                annotations="mean ± sample SD across candidate phrases; leading zero omitted",
                layout=(
                    "4.35 by 3.45 inches; -18-degree row labels; tight outer margins; "
                    "colour bar nearly touches the table"
                ),
                source=[
                    "alignment-extended/summary.csv",
                    "log-decay-directions/summary.csv: forward_log",
                ],
            ),
            indent=2,
        )
        + "\n"
    )
    print(output / "phrase-cosine.png", flush=True)


if __name__ == "__main__":
    main()
