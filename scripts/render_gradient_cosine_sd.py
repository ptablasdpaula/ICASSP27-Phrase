"""Render saved phrase cosine ± SD with explicit colour anchors."""

import csv
import json
from pathlib import Path

import numpy as np
from extend_gradient_cardinality import EXTENDED_COLUMNS
from icassp27_phrase.gradient_assessment import NAMES
from report_gradient_alignment import TEXT_LABELS, compact_cosine, render


def main():
    output = Path("docs/gradient-assessment/alignment-extended")
    with (output / "summary.csv").open() as file:
        rows = {
            (int(r["events"]), r["condition"], r["loss"]): r
            for r in csv.DictReader(file)
            if r["metric"] == "phrase-cosine"
        }
    means, sds = np.zeros((11, 11)), np.zeros((11, 11))
    for i, loss in enumerate(NAMES):
        for j, (events, condition) in enumerate(EXTENDED_COLUMNS):
            row = rows[events, condition, loss]
            means[i, j], sds[i, j] = float(row["mean"]), float(row["candidate_sd"])
    assert np.isfinite(means).all() and np.isfinite(sds).all()
    render(
        output,
        "phrase-cosine",
        means,
        sds,
        EXTENDED_COLUMNS,
        cmap_name="RdBu",
        observed_range=True,
        compact_layout=True,
    )
    headers = [
        f"{ {'joint': 'Joint', 'pitch': 'Pitch', 'time': 'Time'}[c] }: {n}"
        for n, c in EXTENDED_COLUMNS
    ]
    lines = [
        "# Mean whole-phrase cosine ± SD",
        "",
        "| Loss | " + " | ".join(headers) + " |",
        "|---|" + "---:|" * 11,
    ]
    for i, label in enumerate(TEXT_LABELS):
        cells = [f"{compact_cosine(means[i, j])} ± {compact_cosine(sds[i, j])}" for j in range(11)]
        lines.append("| " + " | ".join([label, *cells]) + " |")
    (output / "phrase-cosine.md").write_text("\n".join(lines) + "\n")
    (output / "phrase-cosine-style.json").write_text(
        json.dumps(
            dict(
                cmap="RdBu",
                vmin=float(means.min()),
                observed_min=float(means.min()),
                vmax=1.0,
                ticks=[float(means.min()), 0.5, 1],
                observed_max=float(means.max()),
                annotations="mean ± sample SD across candidate phrases; leading zero omitted",
                layout=(
                    "4.35 by 3.45 inches; 18-degree row labels; tight outer margins; "
                    "colour bar nearly touches the table"
                ),
                source="summary.csv",
            ),
            indent=2,
        )
        + "\n"
    )
    print(output / "phrase-cosine.png", flush=True)


if __name__ == "__main__":
    main()
