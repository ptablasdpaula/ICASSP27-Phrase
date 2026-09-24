"""Regenerate both paper tables from compact numerical results, without optimisation."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
from _recovery_plot import REPORT_LOSSES, render_table
from icassp27_phrase.paths import OUTPUT, REFERENCE, resolve_path, validate_reference
from icassp27_phrase.synth.config import CARDINALITIES


def read_recovery(root):
    with (root / "recovery/per_phrase.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    keys = [(r["loss"], int(r["cardinality"]), int(r["target_index"])) for r in rows]
    expected = {(loss, n, i) for loss in REPORT_LOSSES for n in CARDINALITIES for i in range(150)}
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Incomplete or duplicate recovery results")
    return rows


def generate(root: Path, output: Path):
    if root == REFERENCE:
        validate_reference(root)
    output.mkdir(parents=True, exist_ok=True)
    rows = read_recovery(root)
    medians = {
        m: {
            (loss, n): float(
                np.median(
                    [float(r[m]) for r in rows if r["loss"] == loss and int(r["cardinality"]) == n]
                )
            )
            for loss in REPORT_LOSSES
            for n in CARDINALITIES
        }
        for m in ("pitch_mae_cents", "onset_mae_ms")
    }
    render_table(medians, output / "median_recovery_table.tex")
    timings = {}
    gpu = set()
    for path in sorted((root / "efficiency").glob("*.json")):
        data = json.loads(path.read_text())
        gpu.add(data["gpu"])
        if (
            data["batch"] != 10
            or data["warmup_passes"] != 5
            or data["measured_passes_per_repeat"] != 20
            or not data["candidate_parameters_fixed"]
        ):
            raise ValueError(f"Incompatible efficiency protocol: {path}")
        for loss in data["losses"]:
            if loss in timings:
                raise ValueError(f"Duplicate benchmark: {loss}")
            samples = [r for r in data["rows"] if r["loss"] == loss]
            if len(samples) != 15 or sorted(r["target_begin"] for r in samples) != list(
                range(0, 150, 10)
            ):
                raise ValueError("Incomplete efficiency benchmark")
            timings[loss] = (
                statistics.median(r["mean_ms_per_pass"] for r in samples),
                statistics.median(r["mean_peak_total_mib"] for r in samples) / 1024,
            )
    if len(gpu) != 1:
        raise ValueError("Do not combine GPU models")
    losses = ("single_stft", "mss", "sot_published_composite", "linear_jtfot", "cel")
    if set(timings) != set(losses):
        raise ValueError("Expected all five efficiency losses")
    labels = ("SS", "MSS", "SOT", r"$\mathrm{TF}\mathcal W_2$", r"Ce$\mathcal L$")
    lines = [
        r"\begin{tabular}{@{}lcc@{}}",
        r"\toprule",
        r"Loss & $\downarrow{}$ Time (ms) & $\downarrow{}$ Peak GPU memory (GiB) \\",
        r"\midrule",
    ]
    for name, label in zip(losses, labels, strict=True):
        time, memory = timings[name]
        bt, bm = timings["single_stft"]
        cells = (
            [f"{time:.2f} (---)", f"{memory:.3f} (---)"]
            if name == "single_stft"
            else [
                f"{time:.2f} ($+{100 * (time / bt - 1):.2f}\\%$)",
                f"{memory:.3f} ($+{100 * (memory / bm - 1):.2f}\\%$)",
            ]
        )
        lines.append(" & ".join([label, *cells]) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output / "efficiency_table.tex").write_text("\n".join(lines) + "\n")
    print(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=resolve_path, default=REFERENCE)
    p.add_argument("--output", type=resolve_path, default=OUTPUT / "tables")
    a = p.parse_args()
    generate(a.data, a.output)
