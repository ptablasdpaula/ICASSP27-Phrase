"""Regenerate submitted figures from compact data; no optimisation is run."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from _alignment_plot import render as render_alignment
from _recovery_plot import render_lsd
from icassp27_phrase.data.design import COLUMNS, NAMES
from icassp27_phrase.paths import OUTPUT, REFERENCE, REPO, resolve_path, validate_reference
from make_tables import read_recovery
from render_loss_sweeps import render as render_slices
from run_fixed_gradient_assessment import LABELS


def generate(root: Path, output: Path):
    if root == REFERENCE:
        validate_reference(root)
    output.mkdir(parents=True, exist_ok=True)
    with (root / "gradient-analysis/phrase-cosine.csv").open() as f:
        rows = list(csv.DictReader(f))
    with (root / "gradient-limitations/summary.csv").open() as f:
        extra = list(csv.DictReader(f))
    base = {(r["loss"], int(r["events"]), r["condition"]): r for r in rows}
    more = {(r["loss"], r["condition"]): r for r in extra}
    columns = COLUMNS + ((2, "persistent"), (1, "controls"))
    means = np.empty((len(NAMES), len(columns)))
    sd = np.empty_like(means)
    for i, loss in enumerate(NAMES):
        for j, (n, condition) in enumerate(columns):
            r = (
                base[loss, n, condition]
                if j < len(COLUMNS)
                else more[
                    loss, "persistent_target" if condition == "persistent" else "all_controls"
                ]
            )
            means[i, j] = float(r["mean"])
            sd[i, j] = float(r["candidate_sd"] if j < len(COLUMNS) else r["sample_sd"])
    render_alignment(
        output,
        means,
        sd,
        columns,
        labels=LABELS,
        output_stem="gradient_alignment",
    )
    with np.load(root / "loss-slices/loss_sweeps.npz", allow_pickle=False) as data:
        render_slices(data["displacement"], data["normalised"], output / "loss_sweeps.pdf")
    rows = read_recovery(root)
    lsd = {
        (r["loss"], int(r["cardinality"]), int(r["target_index"])): float(
            r["log_spectral_distance_db"]
        )
        for r in rows
    }
    with (root / "recovery/random_lsd.csv").open() as f:
        for r in csv.DictReader(f):
            lsd["random", int(r["cardinality"]), int(r["target_index"])] = float(
                r["log_spectral_distance_db"]
            )
    render_lsd(lsd, output / "lsd_violin")
    subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/render_cumulative_surfaces.py"),
            "--output-stem",
            str(output / "cumulative_surfaces"),
        ],
        check=True,
    )
    if (output / "Waveguide.pdf").resolve() != (REPO / "paper/figures/Waveguide.pdf").resolve():
        shutil.copyfile(REPO / "paper/figures/Waveguide.pdf", output / "Waveguide.pdf")
    print(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=resolve_path, default=REFERENCE)
    p.add_argument("--output", type=resolve_path, default=OUTPUT / "figures")
    a = p.parse_args()
    generate(a.data, a.output)
