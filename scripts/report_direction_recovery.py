"""Validate all 3,000 fits and publish full-registry recovery summaries."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from icassp27_phrase.config import CARDINALITIES
from icassp27_phrase.targets import load_target
from run_direction_recovery import ROOT, VARIANTS, signature
from test_amplitude_recovery import metrics

OUT = Path("docs/direction-recovery-150")
METRICS = ("pitch_mae_cents", "onset_mae_ms", "log_spectral_distance_db", "joint_event_error")
LABELS = dict(
    orthogonal="Orthogonals",
    orthogonal_lw="Orthogonals + LW",
    clockwise="Clockwise",
    clockwise_lw="Clockwise + LW",
    diagonal="Diagonal reference",
    diagonal_lw="Diagonal reference + LW",
)
ORDER = ("diagonal", "diagonal_lw", *VARIANTS)


def main():
    rows, manifest, compact = [], [], []
    sig = signature()
    for index in range(3000):
        variant = VARIANTS[index // 750]
        events = CARDINALITIES[(index % 750) // 150]
        meta, _ = load_target(events, index % 150 + 1)
        path = ROOT / "raw" / variant / f"{meta.target_id}.json.gz"
        with gzip.open(path, "rt") as f:
            d = json.load(f)
        assert d["signature"] == sig and d["index"] == index and d["variant"] == variant
        assert d["target"]["target_id"] == meta.target_id
        base = d["baseline"]
        assert 0 <= base["updates"] <= 3000
        assert abs(base["best_loss"] - min(s["raw_loss"] for s in base["trajectory"])) < 1e-12
        best = base["best_loss"]
        expected_rotation = variant.startswith("clockwise") and base["stopped_by"] == "patience"
        assert d["rotation_applied"] == expected_rotation
        if expected_rotation:
            trace = d["rotation_trajectory"]
            assert len(trace) == 2401 and trace[-1]["update"] == 2400
            assert np.isfinite([s["raw_loss"] for s in trace]).all()
            best = min(best, min(s["raw_loss"] for s in trace))
        assert abs(d["best_loss"] - best) < 1e-12
        assert d["updates"] == base["updates"] + (2400 if expected_rotation else 0)
        f0, onset = d["best_phrase"]["f0_hz"], d["best_phrase"]["onset_seconds"]
        verified = metrics(f0, onset, meta)
        for key in ("pitch_mae_cents", "onset_mae_ms"):
            assert abs(d["metrics"][key] - verified[key]) < 1e-10
        assert np.isfinite([d["metrics"][key] for key in METRICS]).all()
        if variant.startswith("clockwise"):
            name = "diagonal_lw" if variant.endswith("_lw") else "diagonal"
            rows.append(
                dict(
                    target_id=meta.target_id,
                    events=events,
                    variant=name,
                    **{k: d["baseline_metrics"][k] for k in METRICS},
                    updates=base["updates"],
                    wall_seconds=base["wall_seconds"],
                    rotation_applied=False,
                    best_loss=base["best_loss"],
                    f0_hz=json.dumps(base["best_phrase"]["f0_hz"]),
                    onset_seconds=json.dumps(base["best_phrase"]["onset_seconds"]),
                )
            )
        rows.append(
            dict(
                target_id=meta.target_id,
                events=events,
                variant=variant,
                **{k: d["metrics"][k] for k in METRICS},
                updates=d["updates"],
                wall_seconds=d["wall_seconds"],
                rotation_applied=d["rotation_applied"],
                best_loss=d["best_loss"],
                f0_hz=json.dumps(f0),
                onset_seconds=json.dumps(onset),
            )
        )
        manifest.append(
            dict(index=index, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        )
        d["baseline"].pop("trajectory")
        d.pop("rotation_trajectory", None)
        compact.append(d)
    assert len(rows) == 4500 and len({(r["target_id"], r["variant"]) for r in rows}) == 4500
    with (OUT / "per_phrase.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    groups = defaultdict(list)
    for r in rows:
        groups[r["events"], r["variant"]].append(r)
    summary = []
    for (events, variant), group in groups.items():
        assert len(group) == 150
        for metric in (*METRICS, "updates", "wall_seconds"):
            values = np.array([r[metric] for r in group])
            summary.append(
                dict(
                    events=events,
                    variant=variant,
                    metric=metric,
                    n=150,
                    mean=float(values.mean()),
                    std=float(values.std(ddof=1)),
                    median=float(np.median(values)),
                )
            )
    with (OUT / "summary.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(summary)
    lookup = {(r["target_id"], r["variant"]): r for r in rows}
    paired = []
    for events in CARDINALITIES:
        for variant in ("clockwise", "clockwise_lw"):
            baseline = "diagonal_lw" if variant.endswith("_lw") else "diagonal"
            group = groups[events, variant]
            for metric in METRICS:
                differences = np.array(
                    [r[metric] - lookup[r["target_id"], baseline][metric] for r in group]
                )
                paired.append(
                    dict(
                        events=events,
                        variant=variant,
                        metric=metric,
                        n=150,
                        median_after_minus_before=float(np.median(differences)),
                        mean_after_minus_before=float(differences.mean()),
                        improved=int((differences < -1e-10).sum()),
                        worsened=int((differences > 1e-10).sum()),
                        tied=int((np.abs(differences) <= 1e-10).sum()),
                    )
                )
    with (OUT / "paired_changes.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(paired)
    with gzip.open(OUT / "selected-results.json.gz", "wt") as f:
        json.dump(compact, f, separators=(",", ":"))
    (OUT / "manifest.json").write_text(
        json.dumps(
            dict(
                signature=sig,
                fits=3000,
                target_count=750,
                cardinalities=CARDINALITIES,
                variants=VARIANTS,
                report_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                raw_shards=manifest,
            ),
            indent=2,
        )
        + "\n"
    )
    report(groups, summary, paired)
    print("COMPLETE: all 3,000 fits validated; 4,500 method/reference rows", flush=True)


def report(groups, summary, paired):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = {(r["events"], r["variant"], r["metric"]): r for r in summary}
    lines = [
        "# Full 150-per-cardinality recovery study",
        "",
        "**Complete:** 750 frozen targets × four requested methods = 3,000 fits. "
        "Each method has 150 targets at every event count. Two diagonal-reference rows "
        "come from the initial stages of the clockwise fits, not additional runs.",
        "",
        "[Protocol](protocol.md) · [Per-phrase results](per_phrase.csv) · "
        "[Mean, SD and median](summary.csv) · [Paired changes](paired_changes.csv) · "
        "[Selected states and provenance](selected-results.json.gz)",
        "",
        "**Compute differs:** static orthogonals use the registered ≤3,000-update fitter. "
        "Clockwise adds 2,400 updates only after a diagonal patience stop, with a possible "
        "total of 5,400. The references are pre-rotation results, not matched-budget controls.",
        "",
        "## Median recovery errors",
        "",
        "Each cell is pitch MAE (cents) / onset MAE (ms), medians across 150 phrases.",
        "",
        "| Method | 1 event | 2 events | 4 events | 6 events | 8 events |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in ORDER:
        cells = [
            f"{values[n, variant, 'pitch_mae_cents']['median']:.3f} / "
            f"{values[n, variant, 'onset_mae_ms']['median']:.3f}"
            for n in CARDINALITIES
        ]
        lines.append("| " + LABELS[variant] + " | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Compute and recovery counts",
        "",
        "| Events | Method | Median updates | Median seconds | Rotation applied | "
        "<1 cent and <1 ms |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for n in CARDINALITIES:
        for variant in ORDER:
            group = groups[n, variant]
            lines.append(
                f"| {n} | {LABELS[variant]} | {values[n, variant, 'updates']['median']:.0f} | "
                f"{values[n, variant, 'wall_seconds']['median']:.1f} | "
                f"{sum(r['rotation_applied'] for r in group)}/150 | "
                f"{sum(r['pitch_mae_cents'] < 1 and r['onset_mae_ms'] < 1 for r in group)}/150 |"
            )
    lines += [
        "",
        "![Recovery errors](recovery.png)",
        "",
        "![LSD distributions](lsd.png)",
        "",
        "The full CSV reports mean/sample SD/median. LSD uses the original paper metric, "
        "with its centered STFT and common target-relative amplitude floor. The training "
        "STFT remains uncentered. Joint event error is the registered RMS matched distance.",
        "",
        "Paired changes compare clockwise with its own pre-rotation diagonal fit. Negative "
        "after-minus-before values indicate improvement. These are descriptive comparisons; "
        "no selection of the best loss after seeing results or new significance claims.",
        "",
        "The strict-best training/canonical loss chooses checkpoints, never target parameter "
        "errors. Rotation may retain a state reached before its endpoint. All full trajectories "
        "remain under `results/direction-recovery-150/raw`; the path/hash manifest "
        "identifies every shard, and the compact archive contains all selected states.",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for variant in ORDER:
        for ax, metric in zip(axes, ("pitch_mae_cents", "onset_mae_ms"), strict=True):
            ax.plot(
                CARDINALITIES,
                [max(values[n, variant, metric]["median"], 1e-8) for n in CARDINALITIES],
                marker="o",
                label=LABELS[variant],
            )
            ax.set_yscale("log")
            ax.set_xticks(CARDINALITIES)
            ax.set_xlabel("Events")
            ax.set_ylabel(
                "Median pitch MAE (cents)"
                if metric.startswith("pitch")
                else "Median onset MAE (ms)"
            )
            ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.savefig(OUT / "recovery.png", dpi=170)
    fig.savefig(OUT / "recovery.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 5, figsize=(16, 5), sharey=True, layout="constrained")
    for ax, n in zip(axes, CARDINALITIES, strict=True):
        data = [[r["log_spectral_distance_db"] for r in groups[n, v]] for v in ORDER]
        ax.violinplot(data, showmedians=True, showextrema=True)
        ax.set_xticks(range(1, len(ORDER) + 1), [LABELS[v] for v in ORDER])
        ax.tick_params(axis="x", labelrotation=80, labelsize=8)
        ax.set_title(f"{n} events")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("LSD (dB)")
    fig.savefig(OUT / "lsd.png", dpi=170)
    fig.savefig(OUT / "lsd.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
