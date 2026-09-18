"""Combine saved phrase cosine means with positive summed-dot percentages."""

import csv
from pathlib import Path

import numpy as np
from extend_gradient_cardinality import EXTENDED_COLUMNS
from icassp27_phrase.gradient_assessment import NAMES
from report_gradient_alignment import TEXT_LABELS, compact_cosine, render


def main():
    output = Path("docs/gradient-assessment/alignment-extended")
    with (output / "summary.csv").open() as file:
        rows = {
            (r["metric"], int(r["events"]), r["condition"], r["loss"]): r
            for r in csv.DictReader(file)
        }
    means, percentages = np.zeros((11, 11)), np.zeros((11, 11))
    records = []
    for i, loss in enumerate(NAMES):
        for j, (events, condition) in enumerate(EXTENDED_COLUMNS):
            cosine = rows["phrase-cosine", events, condition, loss]
            positive = rows["phrase-descent", events, condition, loss]
            assert cosine["eligible"] == positive["eligible"]
            means[i, j] = float(cosine["mean"])
            percentages[i, j] = 100 * float(positive["mean"])
            records.append(
                dict(
                    loss=loss,
                    events=events,
                    condition=condition,
                    mean_cosine=means[i, j],
                    positive_phrase_percent=percentages[i, j],
                )
            )
    assert np.isfinite(means).all() and np.isfinite(percentages).all()
    render(
        output,
        "phrase-cosine",
        means,
        np.zeros_like(means),
        EXTENDED_COLUMNS,
        positive_percent=percentages,
    )
    header = [
        f"{ {'joint': 'Joint', 'pitch': 'Pitch', 'time': 'Time'}[c] }: {n}"
        for n, c in EXTENDED_COLUMNS
    ]
    lines = [
        "# Mean phrase cosine (positive phrases %)",
        "",
        "| Loss | " + " | ".join(header) + " |",
        "|---|" + "---:|" * 11,
    ]
    for i, label in enumerate(TEXT_LABELS):
        cells = [f"{compact_cosine(means[i, j])} ({percentages[i, j]:.1f}%)" for j in range(11)]
        lines.append("| " + " | ".join([label, *cells]) + " |")
    lines += [
        "",
        "Colour encodes mean whole-phrase cosine. Parentheses give the percentage of",
        "candidate phrases with a positive summed dot product;",
        "these are not SDs or confidence intervals.",
        "Both measures use the same Hungarian matching and octave–second weighting",
        "as the source tables.",
    ]
    (output / "phrase-cosine-positive.md").write_text("\n".join(lines) + "\n")
    with (output / "phrase-cosine-positive.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(output / "phrase-cosine-positive.png", flush=True)


if __name__ == "__main__":
    main()
