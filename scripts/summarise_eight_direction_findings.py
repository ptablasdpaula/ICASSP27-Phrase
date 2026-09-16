"""Descriptive summaries of the complete screen; no held-out variant selection."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np
from report_eight_directions import label

OUT = Path("docs/eight-direction-screen")
CONDITIONS = (
    ("structured", "isolated_pitch", "pitch_directed"),
    ("structured", "isolated_timing", "onset_directed"),
    ("structured", "isolated_joint", "joint_directed"),
    ("independent", "simultaneous", "pitch_directed"),
    ("independent", "simultaneous", "onset_directed"),
    ("independent", "simultaneous", "joint_directed"),
)


def main():
    with gzip.open(OUT / "summary.csv.gz", "rt") as f:
        rows = [
            r
            for r in csv.DictReader(f)
            if (r["profile"], r["perturbation"], r["metric"]) in CONDITIONS
            and int(r["events"]) in (2, 4, 6, 8)
        ]
    lookup = {
        (int(r["events"]), r["profile"], r["perturbation"], r["metric"], r["variant"]): r
        for r in rows
    }
    names = sorted({r["variant"] for r in rows})
    ranking = []
    for profile, kind, metric in CONDITIONS:
        for name in names:
            values = [float(lookup[n, profile, kind, metric, name]["mean"]) for n in (2, 4, 6, 8)]
            baseline = [
                float(lookup[n, profile, kind, metric, "cel8_015"]["mean"]) for n in (2, 4, 6, 8)
            ]
            ranking.append(
                dict(
                    profile=profile,
                    perturbation=kind,
                    metric=metric,
                    variant=name,
                    directions=label(name),
                    mean_percent=100 * np.mean(values),
                    delta_pp=100 * np.mean(np.array(values) - baseline),
                )
            )
    with (OUT / "rankings.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(ranking[0]))
        writer.writeheader()
        writer.writerows(ranking)
    lines = [
        "# Descriptive findings",
        "",
        "Values below average the target-level percentages equally over **2, 4, 6 and 8 "
        "events**. They summarise multi-event trends, not recovery success. "
        "The full CSV retains each cardinality, target-level sample SD and median. "
        "Pitch-only and onset-only conditions perturb one event while keeping other events "
        "correct; simultaneous errors perturb all events.",
        "",
        "- Orthogonal directions help conditionally, but orthogonal-only losses do not "
        "dominate the diagonal loss. With Log-Weighing, ↑ reaches 100% pitch alignment in "
        "pitch-only slices, versus 99.38% for four diagonals. With simultaneous errors, "
        "the same ↑ loss gives only 62.05% pitch alignment versus 70.31% for the diagonals.",
        "- Adding all four orthogonals improves simultaneous pitch alignment: 69.05% to "
        "71.05% without Log-Weighing, and 70.31% to 72.66% with it. Joint alignment "
        "does not improve (71.75% to 71.50%, and 71.90% to 70.80%, respectively).",
        "- The exploratory mixed subset ↗ ↘ ↑ ↓ + LW is interesting for timing and joint "
        "slices: versus four diagonals + LW, onset-only alignment rises from 90.43% to "
        "93.80%, isolated joint alignment from 89.76% to 92.07%, and simultaneous joint "
        "alignment from 71.90% to 72.65%. Simultaneous pitch alignment falls from 70.31% "
        "to 68.42%. It is a trade-off, not a uniformly better loss.",
        "- These findings support considering different objectives under different error "
        "conditions. They do not establish that an automatic switch can identify those "
        "conditions or improve recovery; the failed-phrase pilots address a different question.",
        "",
        "## Fixed comparison groups",
        "",
        "| Directions | Pitch-only: pitch | Onset-only: onset | Joint slice: joint | "
        "Simultaneous: pitch | Simultaneous: onset | Simultaneous: joint |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for lw in (False, True):
        for mask in (15, 16, 32, 48, 64, 128, 192, 240, 255, 17, 51):
            name = f"cel8_{mask:03d}{'_lw' if lw else ''}"
            values = [
                next(
                    r["mean_percent"]
                    for r in ranking
                    if (r["profile"], r["perturbation"], r["metric"], r["variant"])
                    == (*condition, name)
                )
                for condition in CONDITIONS
            ]
            lines.append(
                "| " + label(name) + " | " + " | ".join(f"{v:.2f}%" for v in values) + " |"
            )
    lines += [
        "",
        "## Exploratory leaders",
        "",
        "Top five configurations per measure, ranked on these same candidates. "
        "Differences are percentage points relative to the uniform four-diagonal loss. "
        "Searching 510 configurations makes these descriptive leaders, not independently "
        "validated choices. Several variants can tie. Other-axis drift and component "
        "alignment should be checked before interpreting a high joint score.",
        "",
    ]
    for profile, kind, metric in CONDITIONS:
        selected = sorted(
            (
                r
                for r in ranking
                if (r["profile"], r["perturbation"], r["metric"]) == (profile, kind, metric)
            ),
            key=lambda r: -r["mean_percent"],
        )[:5]
        lines += [
            f"### {profile} / {kind} / {metric}",
            "",
            "| Subset | Target-directed | Change |",
            "|---|---:|---:|",
        ]
        lines += [
            f"| {r['directions']} | {r['mean_percent']:.2f}% | {r['delta_pp']:+.2f} pp |"
            for r in selected
        ]
        lines.append("")
    lines += [
        "## Reading the results",
        "",
        "All direction subsets average raw directional RMS losses equally. A mixture does "
        "not force equal gradient strength from the two families; the axis-only surfaces "
        "often have smaller scale. Log-Weighing changes spatial quadrature rather than "
        "mass accumulation. No frame-wise or frequency-wise normalisation is introduced.",
        "",
        "The original assignment rule is preserved. In some eight-event timing-only "
        "slices, reassignment creates matched pitch displacement despite unchanged input "
        "pitch coordinates. A target-directed joint gradient is not evidence that every "
        "component points toward its target, nor that optimisation will escape a plateau.",
        "",
    ]
    (OUT / "findings.md").write_text("\n".join(lines))
    print("\n".join(lines[:29]))


if __name__ == "__main__":
    main()
