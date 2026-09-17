"""Validate 4,500 single-stage orthogonal/clockwise/fading fits and publish summaries."""

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
from run_clockwise_recovery import signature as clockwise_signature
from run_direction_recovery import ROOT
from run_direction_recovery import VARIANTS as EXISTING_VARIANTS
from run_direction_recovery import signature as orthogonal_signature
from run_fading_recovery import signature as fading_signature
from test_amplitude_recovery import metrics

VARIANTS = (*EXISTING_VARIANTS, "fading", "fading_lw")

OUT = Path("docs/direction-recovery-150")
METRICS = ("pitch_mae_cents", "onset_mae_ms", "log_spectral_distance_db", "joint_event_error")
LABELS = dict(
    orthogonal="Orthogonals",
    orthogonal_lw="Orthogonals + LW",
    clockwise="Clockwise",
    clockwise_lw="Clockwise + LW",
    fading="Fixed fade 1 s / 1000 Hz",
    fading_lw="Fixed fade + LW",
)


def main():
    rows, manifest, compact = [], [], []
    signatures = dict(
        orthogonal=orthogonal_signature(),
        clockwise=clockwise_signature(),
        fading=fading_signature(),
    )
    for index in range(4500):
        variant = VARIANTS[index // 750]
        rotating = variant.startswith("clockwise")
        fading = variant.startswith("fading")
        family = "fading" if fading else "clockwise" if rotating else "orthogonal"
        events = CARDINALITIES[(index % 750) // 150]
        meta, _ = load_target(events, index % 150 + 1)
        path = ROOT / "raw" / variant / f"{meta.target_id}.json.gz"
        with gzip.open(path, "rt") as f:
            d = json.load(f)
        assert d["signature"] == signatures[family]
        assert d["index"] == index and d["variant"] == variant
        assert d["target"]["target_id"] == meta.target_id
        key = "fit" if rotating or fading else "baseline"
        result = d[key]
        if fading:
            assert d["schema"] == "fixed-fading-v1" and not d["fade_schedule"]
            assert d["time_horizon_seconds"] == 1 and d["frequency_horizon_hz"] == 1000
            assert d["log_weighing"] == variant.endswith("_lw")
        elif rotating:
            assert d["schema"] == "single-stage-clockwise-v1"
            assert not d["diagonal_warmup"] and not d["refinement"] and "baseline" not in d
        else:
            assert not d["rotation_applied"] and "rotation_trajectory" not in d
        assert d["updates"] == result["updates"] <= 3000
        values = [s["raw_loss"] for s in result["trajectory"]]
        assert np.isfinite(values).all()
        assert abs(result["best_loss"] - min(values)) < 1e-12
        assert abs(d["best_loss"] - result["best_loss"]) < 1e-12
        f0, onset = d["best_phrase"]["f0_hz"], d["best_phrase"]["onset_seconds"]
        verified = metrics(f0, onset, meta)
        for metric in ("pitch_mae_cents", "onset_mae_ms"):
            assert abs(d["metrics"][metric] - verified[metric]) < 1e-10
        assert np.isfinite([d["metrics"][metric] for metric in METRICS]).all()
        rows.append(
            dict(
                target_id=meta.target_id,
                events=events,
                variant=variant,
                **{k: d["metrics"][k] for k in METRICS},
                updates=d["updates"],
                wall_seconds=d["wall_seconds"],
                stopped_by=result["stopped_by"],
                best_loss=d["best_loss"],
                f0_hz=json.dumps(f0),
                onset_seconds=json.dumps(onset),
            )
        )
        manifest.append(
            dict(index=index, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        )
        d[key].pop("trajectory")
        compact.append(d)
    assert len(rows) == len({(r["target_id"], r["variant"]) for r in rows}) == 4500
    write_csv("per_phrase.csv", rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["events"], row["variant"]].append(row)
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
    write_csv("summary.csv", summary)
    lookup = {(r["target_id"], r["variant"]): r for r in rows}
    paired = []
    for events in CARDINALITIES:
        for variant, reference in (
            ("clockwise", "orthogonal"),
            ("clockwise_lw", "orthogonal_lw"),
            ("fading", "orthogonal"),
            ("fading_lw", "orthogonal_lw"),
            ("fading", "clockwise"),
            ("fading_lw", "clockwise_lw"),
            ("fading_lw", "fading"),
        ):
            for metric in METRICS:
                differences = np.array(
                    [
                        r[metric] - lookup[r["target_id"], reference][metric]
                        for r in groups[events, variant]
                    ]
                )
                paired.append(
                    dict(
                        events=events,
                        variant=variant,
                        reference=reference,
                        metric=metric,
                        n=150,
                        median_variant_minus_reference=float(np.median(differences)),
                        mean_variant_minus_reference=float(differences.mean()),
                        variant_better=int((differences < -1e-10).sum()),
                        variant_worse=int((differences > 1e-10).sum()),
                        tied=int((np.abs(differences) <= 1e-10).sum()),
                    )
                )
    write_csv("paired_changes.csv", paired)
    with gzip.open(OUT / "selected-results.json.gz", "wt") as f:
        json.dump(compact, f, separators=(",", ":"))
    (OUT / "manifest.json").write_text(
        json.dumps(
            dict(
                signatures=signatures,
                fits=4500,
                target_count=750,
                cardinalities=CARDINALITIES,
                variants=VARIANTS,
                schema="single-stage-direction-recovery-v3",
                report_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                raw_shards=manifest,
            ),
            indent=2,
        )
        + "\n"
    )
    report(groups, summary)
    print("COMPLETE: all 4,500 single-stage fits validated", flush=True)


def write_csv(name, rows):
    with (OUT / name).open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def report(groups, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = {(r["events"], r["variant"], r["metric"]): r for r in summary}
    lines = [
        "# Full orthogonal / clockwise / fixed-fading recovery",
        "",
        "**Complete:** 750 frozen targets × six methods = 4,500 fits. "
        "Each method has 150 targets at each of 1/2/4/6/8 events.",
        "",
        "[Protocol](protocol.md) · [Per-phrase results](per_phrase.csv) · "
        "[Mean, SD and median](summary.csv) · [Paired comparisons](paired_changes.csv) · "
        "[Selected states](selected-results.json.gz)",
        "",
        "All methods start from the standard initial guess and use the same Adam, "
        "patience, rollback and ≤3,000-update cap. Clockwise rotates throughout, without "
        "a diagonal pre-fit, warm-up or refinement. Actual updates and runtime may differ.",
        "",
        "For clockwise fits, gradients use the rotating loss; the fixed eight-direction "
        "average with matching Log-Weighing controls patience and checkpoint selection. "
        "Orthogonal fits monitor their static four-direction objective. The monitor is "
        "not used for clockwise backward updates.",
        "",
        "Fixed fading uses all four diagonal directions, with logarithmic fade-to-zero "
        "horizons of 1 second and 1000 Hz throughout. Uniform and Log-Weighed variants "
        "use their own fixed training loss for patience and checkpoint selection.",
        "",
        "## Median recovery errors",
        "",
        "Pitch MAE (cents) / onset MAE (ms), medians across 150 phrases.",
        "",
        "| Method | 1 event | 2 events | 4 events | 6 events | 8 events |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
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
        "| Events | Method | Median updates | Median seconds | Patience stops | "
        "<1 cent and <1 ms |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for n in CARDINALITIES:
        for variant in VARIANTS:
            group = groups[n, variant]
            lines.append(
                f"| {n} | {LABELS[variant]} | {values[n, variant, 'updates']['median']:.0f} | "
                f"{values[n, variant, 'wall_seconds']['median']:.1f} | "
                f"{sum(r['stopped_by'] == 'patience' for r in group)}/150 | "
                f"{sum(r['pitch_mae_cents'] < 1 and r['onset_mae_ms'] < 1 for r in group)}/150 |"
            )
    lines += [
        "",
        "![Recovery errors](recovery.png)",
        "",
        "![LSD distributions](lsd.png)",
        "",
        "Paired differences compare each named variant and reference on the same targets. "
        "Negative variant-minus-reference differences favour the variant. "
        "These are descriptive comparisons, not new significance claims.",
        "",
        "The full CSV includes mean/sample SD/median. LSD exactly follows the original "
        "paper metric. Joint event error is the registered RMS matched distance. "
        "Ground-truth errors never select checkpoints. Full trajectories remain under "
        "`results/direction-recovery-150/raw`; the manifest hashes every accepted shard. "
        "Superseded staged-clockwise outputs are excluded.",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for variant in VARIANTS:
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
        data = [[r["log_spectral_distance_db"] for r in groups[n, v]] for v in VARIANTS]
        ax.violinplot(data, showmedians=True, showextrema=True)
        ax.set_xticks(range(1, len(VARIANTS) + 1), [LABELS[v] for v in VARIANTS])
        ax.tick_params(axis="x", labelrotation=80, labelsize=8)
        ax.set_title(f"{n} events")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("LSD (dB)")
    fig.savefig(OUT / "lsd.png", dpi=170)
    fig.savefig(OUT / "lsd.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
