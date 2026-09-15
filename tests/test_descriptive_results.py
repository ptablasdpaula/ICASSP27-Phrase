from __future__ import annotations

import math
from pathlib import Path

import pytest

from scripts.render_descriptive_results import (
    CARDINALITIES,
    LOSS_NAMES,
    format_descriptive_cell,
    format_table_value,
    render_table,
)


def test_descriptive_value_format_keeps_registered_display_floor() -> None:
    assert format_table_value(0.00999) == "<.01"
    assert format_table_value(0.01) == "0.01"
    assert format_table_value(123.456) == "123"
    assert format_table_value(1031.768) == "1030"
    assert format_descriptive_cell(3.0, math.sqrt(7.0), 2.0) == (
        r"\mbox{\ensuremath{3\mathord{\pm}2.65(2)}}"
    )


def test_table_reports_mean_sample_sd_and_median_without_ranking(
    tmp_path: Path,
) -> None:
    rows = []
    for loss in LOSS_NAMES:
        for cardinality in CARDINALITIES:
            for target_index, value in enumerate((1.0, 2.0, 6.0)):
                rows.append(
                    {
                        "loss": loss,
                        "cardinality": str(cardinality),
                        "target_index": str(target_index),
                        "pitch_mae_cents": str(value),
                        "onset_mae_ms": str(value),
                    }
                )

    output = tmp_path / "recovery_table.tex"
    report = render_table(rows, output)
    first = report["statistics"]["pitch_mae_cents"][LOSS_NAMES[0]]["1"]
    assert first == pytest.approx(
        {
            "mean": 3.0,
            "sample_standard_deviation": math.sqrt(7.0),
            "median": 2.0,
        }
    )
    assert report["display"] == "mean +/- sample standard deviation (median)"

    source = output.read_text(encoding="utf-8")
    assert r"\mbox{\ensuremath{3\mathord{\pm}2.65(2)}}" in source
    assert r"\mathbf" not in source
    assert r"\underline" not in source
    assert r"\mathit" not in source
