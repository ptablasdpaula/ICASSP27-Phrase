"""Render the saved phrase cosine table with Matplotlib's Magma map."""

import csv
import json
from pathlib import Path

import numpy as np
from extend_gradient_cardinality import EXTENDED_COLUMNS
from icassp27_phrase.gradient_assessment import NAMES
from report_gradient_alignment import render


def main():
    output = Path("docs/gradient-assessment/alignment-extended")
    with (output / "summary.csv").open() as file:
        rows = {
            (int(row["events"]), row["condition"], row["loss"]): row
            for row in csv.DictReader(file)
            if row["metric"] == "phrase-cosine"
        }
    means, sds = np.zeros((11, 11)), np.zeros((11, 11))
    for loss_index, loss in enumerate(NAMES):
        for column, (events, condition) in enumerate(EXTENDED_COLUMNS):
            row = rows[events, condition, loss]
            means[loss_index, column] = float(row["mean"])
            sds[loss_index, column] = float(row["candidate_sd"])
    assert np.isfinite(means).all() and np.isfinite(sds).all()
    render(
        output,
        "phrase-cosine",
        means,
        sds,
        EXTENDED_COLUMNS,
        cmap_name="magma",
        output_stem="phrase-cosine-magma",
    )
    (output / "phrase-cosine-magma-style.json").write_text(
        json.dumps(
            {
                "cmap": "magma",
                "vmin": -1.0,
                "vmax": 1.0,
                "ticks": [-1, -0.5, 0, 0.5, 1],
                "annotations": "mean ± sample SD; leading zero omitted",
                "source": "summary.csv",
            },
            indent=2,
        )
        + "\n"
    )
    print(output / "phrase-cosine-magma.png", flush=True)


if __name__ == "__main__":
    main()
