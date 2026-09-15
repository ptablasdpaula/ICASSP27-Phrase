#!/usr/bin/env python3
"""Compute the paper's exploratory TFW2-versus-BiCuL family analysis."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from render_descriptive_results import (
    CARDINALITIES,
    atomic_json,
    load_rows,
    read_signed_json,
    sha256,
)
from scipy.stats import wilcoxon

FAMILIES = {
    "TFW2": ("linear_jtfot", "log_jtfot"),
    "BiCuL": (
        "bidirectional_cumulative_energy",
        "log_quadrature_bicul",
    ),
}
OUTCOMES = (
    "pitch_mae_cents",
    "onset_mae_ms",
    "log_spectral_distance_db",
)


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm-adjusted p-values in the input order."""
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    adjusted = [0.0] * count
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (count - rank) * p_values[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted


def analyze(rows: list[dict[str, str]]) -> dict[str, Any]:
    values = {
        (int(row["cardinality"]), int(row["target_index"]), row["loss"]): {
            outcome: float(row[outcome]) for outcome in OUTCOMES
        }
        for row in rows
    }

    tests: list[dict[str, Any]] = []
    for cardinality in CARDINALITIES:
        for outcome in OUTCOMES:
            family_scores: dict[str, np.ndarray] = {}
            for family, losses in FAMILIES.items():
                family_scores[family] = np.asarray(
                    [
                        np.mean(
                            [
                                values[(cardinality, target_index, loss)][outcome]
                                for loss in losses
                            ],
                            dtype=np.float64,
                        )
                        for target_index in range(150)
                    ],
                    dtype=np.float64,
                )

            difference = family_scores["BiCuL"] - family_scores["TFW2"]
            result = wilcoxon(
                difference,
                alternative="two-sided",
                zero_method="wilcox",
                correction=False,
                method="asymptotic",
            )
            tests.append(
                {
                    "cardinality": cardinality,
                    "outcome": outcome,
                    "paired_phrases": int(difference.size),
                    "tfw2_family_median": float(np.median(family_scores["TFW2"])),
                    "bicul_family_median": float(np.median(family_scores["BiCuL"])),
                    "median_paired_difference": float(np.median(difference)),
                    "bicul_lower_count": int(np.count_nonzero(difference < 0.0)),
                    "tie_count": int(np.count_nonzero(difference == 0.0)),
                    "wilcoxon_statistic": float(result.statistic),
                    "unadjusted_p": float(result.pvalue),
                }
            )

    adjusted = holm_adjust([test["unadjusted_p"] for test in tests])
    for test, adjusted_p in zip(tests, adjusted, strict=True):
        test["holm_adjusted_p"] = adjusted_p
        test["below_exploratory_0.05_threshold"] = adjusted_p < 0.05

    return {
        "analysis": "post-hoc exploratory loss-family comparison",
        "families": FAMILIES,
        "family_score": (
            "arithmetic mean of the two family variants within each phrase "
            "and outcome"
        ),
        "difference_direction": "BiCuL family minus TFW2 family; negative favours BiCuL",
        "pairing": "exact shared target within event count",
        "test": "two-sided paired Wilcoxon signed-rank",
        "wilcoxon_zero_method": "discard zero differences",
        "wilcoxon_p_method": "asymptotic normal approximation without continuity correction",
        "multiplicity_control": "Holm adjustment across all 15 outcome-by-event-count comparisons",
        "exploratory_threshold": 0.05,
        "comparison_count": len(tests),
        "below_threshold_count": sum(
            test["below_exploratory_0.05_threshold"] for test in tests
        ),
        "tests": tests,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-phrase", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = read_signed_json(args.summary, status="complete")
    rows = load_rows(args.per_phrase, summary, milestone_n=150)
    payload = {
        "schema": "exploratory-loss-family-tests-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "scipy_version": scipy.__version__,
        "sources": {
            "per_phrase_csv": str(args.per_phrase),
            "per_phrase_csv_sha256": sha256(args.per_phrase),
            "report": str(args.summary),
            "report_sha256": sha256(args.summary),
            "report_payload_sha256": summary["payload_sha256"],
        },
        **analyze(rows),
    }
    atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
