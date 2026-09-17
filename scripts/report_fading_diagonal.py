"""Summarise the frozen eight-event fading-memory diagonal pilot."""

from __future__ import annotations

import csv
import gzip
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from test_fading_diagonal import DOCS, ROOT, SETTINGS, signature


def main():
    results, rows = [], []
    for index in range(10):
        with gzip.open(ROOT / f"{index:02d}.json.gz", "rt") as handle:
            result = json.load(handle)
        assert result["index"] == index and result["signature"] == signature()
        assert result["time_horizon_seconds"] == SETTINGS[index % 5][0]
        fit = result["fit"]
        assert fit["updates"] <= 3000
        assert abs(fit["best_loss"] - min(s["raw_loss"] for s in fit["trajectory"])) < 1e-12
        results.append(result)
        rows.append(
            dict(
                index=index,
                mode=result["mode"],
                time_horizon_seconds=result["time_horizon_seconds"],
                frequency_horizon_hz=result["frequency_horizon_hz"],
                **{
                    k: result["metrics"][k]
                    for k in (
                        "pitch_mae_cents",
                        "onset_mae_ms",
                        "joint_event_error",
                        "log_spectral_distance_db",
                    )
                },
                canonical_loss=result["final_canonical_loss"],
                training_loss=fit["best_loss"],
                updates=fit["updates"],
                stopped_by=fit["stopped_by"],
                wall_seconds=fit["wall_seconds"],
            )
        )
    with (DOCS / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with gzip.open(DOCS / "raw-results.json.gz", "wt") as handle:
        json.dump(results, handle, separators=(",", ":"))
    labels = ["Ordinary", "2 s / 2 kHz", "1 s / 1 kHz", ".5 s / .5 kHz", ".25 s / .25 kHz"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    for row, mode in enumerate(("plateau", "fresh")):
        for col, (metric, title) in enumerate(
            (("pitch_mae_cents", "Pitch MAE (cents)"), ("onset_mae_ms", "Onset MAE (ms)"))
        ):
            ax = axes[row, col]
            values = [r[metric] for r in rows if r["mode"] == mode]
            ax.bar(np.arange(5), values)
            ax.set_xticks(np.arange(5), labels, rotation=20, ha="right")
            ax.set_title(f"{mode.title()} start — {title}")
            ax.grid(axis="y", alpha=0.25)
    fig.savefig(DOCS / "comparison.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    distance = np.linspace(0, 2, 600)
    for horizon in (2, 1, 0.5, 0.25):
        weight = np.maximum(0, 1 - np.log1p(9 * distance / horizon) / np.log(10))
        ax.plot(distance, weight, label=f"H={horizon:g} s")
    ax.axhline(1, color="black", linestyle="--", label="Ordinary accumulation")
    ax.set(xlabel="Distance from contribution (s)", ylabel="Retained weight", ylim=(-0.02, 1.05))
    ax.legend()
    fig.savefig(DOCS / "decay.png", dpi=160)
    plt.close(fig)
    initial = results[0]["initial_metrics"]
    lines = [
        "# Logarithmically fading diagonal accumulation",
        "",
        "Complete: one frozen eight-event failure, C08-T0000; "
        "five objectives × two initial states.",
        "",
        "[Protocol](protocol.md) · [Summary CSV](summary.csv) · "
        "[Full trajectories](raw-results.json.gz)",
        "",
        f"Saved plateau: {initial['pitch_mae_cents']:.3f} cents pitch MAE, "
        f"{initial['onset_mae_ms']:.3f} ms onset MAE.",
        "",
        "All fits use all four diagonal directions, fixed amplitudes, no Log-Weighing, "
        "the registered Adam/patience schedule and at most 3,000 updates. Fading is "
        "applied to each individual contribution along both axes. Each fit selects "
        "its best own-objective iterate; target parameter errors never select checkpoints.",
        "",
        "| Start | Fade horizon (time / frequency) | Pitch MAE (cents) | "
        "Onset MAE (ms) | Joint RMS error | Updates |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['mode']} | {labels[r['index'] % 5]} | {r['pitch_mae_cents']:.3f} | "
            f"{r['onset_mae_ms']:.3f} | {r['joint_event_error']:.6f} | {r['updates']} |"
        )
    lines += [
        "",
        "![Recovery errors](comparison.png)",
        "",
        "![Fade curves](decay.png)",
        "",
        "These are single-case exploratory results, with four predeclared decay scales. "
        "Faded training-loss magnitudes are not comparable to the ordinary loss. "
        "The CSV also reports the same unfaded diagonal loss and LSD for every selected "
        "state. The joint error is the registered RMS matched error. The saved plateau "
        "and fresh starts answer different questions; this is not a multi-phrase validation.",
        "",
    ]
    (DOCS / "README.md").write_text("\n".join(lines))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
