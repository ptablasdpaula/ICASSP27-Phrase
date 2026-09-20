#!/usr/bin/env python3
"""Append the two limitation diagnostics to the paper gradient heatmap."""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fixed_gradient_assessment import COLUMNS, NAMES  # noqa: E402
from run_fixed_gradient_assessment import LABELS  # noqa: E402
from report_gradient_alignment import render  # noqa: E402

ROOT = Path(__file__).parents[1]
BASE = ROOT / "docs/gradient-assessment/16k/phrase-cosine.csv"
EXTRA = ROOT / "docs/gradient-assessment/limitations/summary.csv"
OUTPUT = ROOT / "docs/gradient-assessment/limitations"
PAPER_FIGURE = ROOT / "paper/figures/gradient_alignment.pdf"
EXTRA_COLUMNS = ((2, "persistent"), (1, "controls"))


def main() -> None:
    with BASE.open() as file:
        base_rows = list(csv.DictReader(file))
    with EXTRA.open() as file:
        extra_rows = list(csv.DictReader(file))
    base = {
        (row["loss"], int(row["events"]), row["condition"]): row
        for row in base_rows
        if row["metric"] == "phrase-cosine"
    }
    extra = {(row["loss"], row["condition"]): row for row in extra_rows}
    columns = COLUMNS + EXTRA_COLUMNS
    means = np.zeros((len(NAMES), len(columns)))
    deviations = np.zeros_like(means)
    for loss_index, loss in enumerate(NAMES):
        for column, (events, condition) in enumerate(COLUMNS):
            row = base[(loss, events, condition)]
            means[loss_index, column] = float(row["mean"])
            deviations[loss_index, column] = float(row["candidate_sd"])
        for column, condition in enumerate(
            ("persistent_target", "all_controls"), start=len(COLUMNS)
        ):
            row = extra[(loss, condition)]
            means[loss_index, column] = float(row["mean"])
            deviations[loss_index, column] = float(row["sample_sd"])
    render(
        OUTPUT,
        "phrase-cosine",
        means,
        deviations,
        columns=columns,
        cmap_name="RdBu",
        cropped_colorbar=True,
        compact_layout=True,
        labels=LABELS,
        output_stem="gradient-alignment-extended",
    )
    shutil.copyfile(OUTPUT / "gradient-alignment-extended.pdf", PAPER_FIGURE)
    print(PAPER_FIGURE)


if __name__ == "__main__":
    main()
