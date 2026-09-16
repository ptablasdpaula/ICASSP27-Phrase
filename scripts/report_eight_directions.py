"""Aggregate the complete eight-direction screen with the original scoring rules."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np
from explore_eight_directions import write_explorer
from icassp27_phrase.cel_screen import design, score
from screen_eight_directions import DIRECTIONS, NAMES, signature

ROOT = Path("results/eight-direction-screen/raw")
OUT = Path("docs/eight-direction-screen")
ARROWS = ("↗", "↘", "↖", "↙", "↑", "↓", "→", "←")
COUNTS = (1, 2, 4, 6, 8)
METRICS = ("pitch_directed", "onset_directed", "joint_directed")
CONDITIONS = (
    ("structured", "isolated_pitch"),
    ("structured", "isolated_timing"),
    ("structured", "isolated_joint"),
    ("independent", "simultaneous"),
)


def label(name):
    mask = int(name.split("_")[1])
    return " ".join(a for i, a in enumerate(ARROWS) if mask & (1 << i)) + (
        " + LW" if name.endswith("_lw") else ""
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    groups = {}
    metadata = []
    total, ties, nonfinite = 0, 0, 0
    for target in design():
        path = ROOT / f"{target.name}.npz"
        with np.load(path) as archive:
            data = dict(archive)
        assert str(data["signature"]) == signature()
        delta = data["target"][data["assignments"]] - data["candidates"]
        # Build subsets in small blocks to bound peak RAM on eight-event targets.
        batches = {}
        for start in range(0, len(NAMES), 32):
            gradients = []
            for name in NAMES[start : start + 32]:
                mask = int(name.split("_")[1])
                indices = [
                    i + (8 if name.endswith("_lw") else 0) for i in range(8) if mask & (1 << i)
                ]
                gradients.append(data["elementary_gradients"][:, indices].mean(axis=1))
            for metric, value in score(np.stack(gradients, axis=1), delta).items():
                batches.setdefault(metric, []).append(value)
        scores = {key: np.concatenate(value, axis=1) for key, value in batches.items()}
        total += len(delta)
        ties += int(data["tied"].sum())
        nonfinite += int(scores.pop("nonfinite").sum())
        for kind in np.unique(data["kinds"]):
            selected = (data["kinds"] == kind) & ~data["tied"]
            for metric, values in scores.items():
                count = np.isfinite(values[selected]).sum(axis=0)
                means = np.divide(
                    np.nansum(values[selected], axis=0),
                    count,
                    out=np.full(len(NAMES), np.nan),
                    where=count > 0,
                )
                for profile in {
                    target.profile,
                    "independent" if target.profile == "independent" else "structured",
                }:
                    key = (len(target.coordinates), profile, str(kind), metric)
                    groups.setdefault(key, []).append(means)
        metadata.append(
            dict(
                target=target.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                old_sha256=str(data["old_sha256"]),
            )
        )
    rows = []
    with gzip.open(OUT / "summary.csv.gz", "wt") as f:
        writer = None
        for key, values in sorted(groups.items()):
            values = np.stack(values)
            # Paired target-level comparison to uniform four-diagonal reference.
            differences = values - values[:, [14]]
            for i, name in enumerate(NAMES):
                finite = values[:, i][np.isfinite(values[:, i])]
                diff = differences[:, i][np.isfinite(differences[:, i])]
                row = dict(
                    events=key[0],
                    profile=key[1],
                    perturbation=key[2],
                    metric=key[3],
                    variant=name,
                    directions=label(name),
                    phrases=len(finite),
                    mean=float(finite.mean()) if len(finite) else None,
                    std=float(finite.std(ddof=1)) if len(finite) > 1 else None,
                    median=float(np.median(finite)) if len(finite) else None,
                    paired_mean_difference=float(diff.mean()) if len(diff) else None,
                    paired_std_difference=float(diff.std(ddof=1)) if len(diff) > 1 else None,
                )
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                if int(name.split("_")[1]) <= 15 or (
                    (key[1], key[2]) in CONDITIONS and key[3] in METRICS
                ):
                    rows.append(row)
    lookup = {
        (r["events"], r["profile"], r["perturbation"], r["metric"], r["variant"]): r for r in rows
    }
    # Every old diagonal-only result must reproduce the published table.
    max_difference = 0.0
    with Path("docs/cel-gradient-screen/summary.csv").open() as f:
        for old in csv.DictReader(f):
            name = f"cel8_{int(old['variant'].split('_')[1]):03d}" + (
                "_lw" if old["variant"].endswith("_lw") else ""
            )
            new = lookup[
                int(old["events"]), old["profile"], old["perturbation"], old["metric"], name
            ]
            assert int(old["phrases"]) == new["phrases"]
            for metric in ("mean", "std", "median"):
                if old[metric]:
                    difference = abs(float(old[metric]) - new[metric])
                    assert difference < 1e-12
                    max_difference = max(max_difference, difference)
                else:
                    assert new[metric] is None
    provenance = dict(
        signature=signature(),
        report_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        numpy_version=np.__version__,
        directions=DIRECTIONS,
        variants=NAMES,
        candidates=total,
        tied_assignments=ties,
        nonfinite=nonfinite,
        old_diagonal_summary_max_difference=max_difference,
        shards=metadata,
        aggregation="candidate means within target, then equal-weight targets; sample SD",
        backend="CPU float64, compiled TorchLPC",
        quadrature="oriented active-axis widths; trapezoidal inactive-axis widths",
        subset_rule="equal mean of selected elementary losses; no family rescaling",
    )
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    with zipfile.ZipFile(OUT / "raw-results.zip", "w", compression=zipfile.ZIP_STORED) as z:
        for target in design():
            z.write(ROOT / f"{target.name}.npz", f"{target.name}.npz")
    summaries(lookup)
    write_explorer(lookup, OUT, NAMES, label)
    print(json.dumps({k: v for k, v in provenance.items() if k not in ("shards", "variants")}))


def summaries(lookup):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    selected = [1, 2, 4, 8, 15, 16, 32, 48, 64, 128, 192, 240, 255, 17, 33, 65, 129]
    names = [f"cel8_{mask:03d}{'_lw' if lw else ''}" for lw in (False, True) for mask in selected]
    text = [
        "# Eight-direction CeL gradient screen",
        "",
        "All 255 nonempty subsets, each uniform and with Log-Weighing (510 configurations). "
        "Same 173 targets and 13,099 candidate phrases as the earlier diagonal screen. "
        "This is a local-gradient screen, not a gradient-descent experiment.",
        "",
        "[Findings](findings.md) · [Protocol](protocol.md) · "
        "[Full CSV, compressed](summary.csv.gz) · "
        "[Interactive table](explorer.html) · [All-subset plots](all-subsets.pdf) · "
        "[Raw arrays](raw-results.zip)",
        "",
        "Each entry below is pitch / onset / joint target-directed event percentage. "
        "Targets are weighted equally within each condition and event count. "
        "A dash means no displaced coordinates eligible for that statistic. "
        "Full CSV includes sample SD, median, cosine alignment and correct-coordinate drift.",
        "",
    ]
    for profile, kind in CONDITIONS:

        def matrix(variants, metric, profile=profile, kind=kind):
            return (
                np.array(
                    [
                        [lookup[n, profile, kind, metric, name]["mean"] for n in COUNTS]
                        for name in variants
                    ],
                    dtype=float,
                )
                * 100
            )

        text += [
            f"## {profile}: {kind}",
            "",
            f"![Selected subsets]({profile}-{kind}.png)",
            "",
            "| Subset | 1 event | 2 events | 4 events | 6 events | 8 events |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name in names:
            cells = []
            for n in COUNTS:
                values = [lookup[n, profile, kind, metric, name]["mean"] for metric in METRICS]
                cells.append(" / ".join("—" if v is None else f"{v * 100:.1f}" for v in values))
            text.append("| " + label(name) + " | " + " | ".join(cells) + " |")
        fig, axes = plt.subplots(1, 3, figsize=(12, 12), sharey=True, layout="constrained")
        for ax, metric in zip(axes, METRICS, strict=True):
            im = ax.imshow(matrix(names, metric), vmin=0, vmax=100, aspect="auto")
            ax.set_xticks(range(5), COUNTS)
            ax.set_yticks(range(len(names)), [label(n) for n in names], fontsize=8)
            ax.set_title(metric.replace("_", " "))
            ax.set_xlabel("Events")
        fig.colorbar(im, ax=axes, label="Target-directed events (%)", shrink=0.5)
        fig.suptitle(f"{profile}: {kind}")
        fig.savefig(OUT / f"{profile}-{kind}.png", dpi=140)
        plt.close(fig)
    text += [
        "",
        "## Interpretation",
        "",
        "Mixed subsets use the raw elementary RMS values with equal weights; "
        "orthogonal and diagonal families are not separately normalised. "
        "Up/down accumulation stays within each frame; right/left stays within each frequency bin. "
        "All surfaces use total target power and the same square-root feature.",
        "",
        "Matching and eligibility follow the original Hungarian rule. In eight-event "
        "structured timing-only slices, reassignment can introduce matched pitch error. "
        "Joint alignment can hide a wrong pitch or onset component. These are derivatives in "
        "normalised physical coordinates, not Adam updates or convergence guarantees. "
        "Comparisons across 510 configurations are exploratory, without held-out selection.",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(text))
    # Pagination makes all 510 subsets inspectable without an unreadable giant image.
    with PdfPages(OUT / "all-subsets.pdf") as pdf:
        for profile, kind in CONDITIONS:
            for start in range(0, len(NAMES), 34):
                variants = NAMES[start : start + 34]
                fig, axes = plt.subplots(1, 3, figsize=(12, 12), sharey=True, layout="constrained")
                for ax, metric in zip(axes, METRICS, strict=True):
                    values = (
                        np.array(
                            [
                                [lookup[n, profile, kind, metric, name]["mean"] for n in COUNTS]
                                for name in variants
                            ],
                            dtype=float,
                        )
                        * 100
                    )
                    im = ax.imshow(values, vmin=0, vmax=100, aspect="auto")
                    ax.set_xticks(range(5), COUNTS)
                    ax.set_yticks(range(len(variants)), [label(n) for n in variants], fontsize=7)
                    ax.set_xlabel("Events")
                    ax.set_title(metric.replace("_", " "))
                fig.colorbar(im, ax=axes, label="Target-directed events (%)", shrink=0.5)
                fig.suptitle(
                    f"{profile}: {kind}; configurations {start + 1}–{start + len(variants)}"
                )
                pdf.savefig(fig)
                plt.close(fig)


if __name__ == "__main__":
    main()
