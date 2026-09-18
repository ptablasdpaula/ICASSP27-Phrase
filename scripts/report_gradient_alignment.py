"""Rescore frozen GPU gradients for event and whole-phrase alignment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from assess_gradients import checked_data, save_json, scores_from, shard_path, write_csv
from icassp27_phrase.gradient_assessment import COLUMNS, NAMES, signature, targets

LABELS = [
    r"$L_1$",
    r"$L_2$",
    "Single STFT",
    "Linear MSS",
    "Smooth MSS",
    "SOT",
    r"$\mathrm{TF}\mathcal{W}_2$",
    r"log-$\mathrm{TF}\mathcal{W}_2$",
    r"Ce$\mathcal{L}$",
    r"Log-Ce$\mathcal{L}$",
    r"Fade-Ce$\mathcal{L}$",
]
TEXT_LABELS = [
    "L1",
    "L2",
    "Single STFT",
    "Linear MSS",
    "Smooth MSS",
    "SOT",
    "TFW2",
    "log-TFW2",
    "CeL",
    "Log-CeL",
    "Fade-CeL",
]
METRICS = {
    "event-cosine": "Mean event cosine",
    "phrase-descent": "Whole-phrase descent (%)",
    "phrase-cosine": "Mean whole-phrase cosine",
}


def alignment(data):
    """Return candidate × loss arrays, preserving archived exclusion conventions."""
    event = scores_from(data)["cosine"]
    descent = -data["gradients"]
    delta = (data["target"][data["assignments"]] - data["candidates"])[:, None]
    dot = (descent * delta).sum(axis=(-2, -1))
    norms = np.sqrt((descent**2).sum(axis=(-2, -1))) * np.sqrt((delta**2).sum(axis=(-2, -1)))
    cosine = np.divide(dot, norms, out=np.zeros_like(dot), where=norms > 0)
    cosine = np.clip(cosine, -1, 1)
    valid = ~data["ties"] & (np.abs(delta[:, 0]) > 1e-12).any(axis=(-2, -1))
    positive = (dot > 0).astype(float)
    cosine[~valid] = np.nan
    positive[~valid] = np.nan
    np.testing.assert_array_equal(cosine[valid] > 0, positive[valid] > 0)
    if data["target"].shape[0] == 1:
        np.testing.assert_allclose(event, cosine, atol=1e-14, equal_nan=True)
    return {"event-cosine": event, "phrase-descent": positive, "phrase-cosine": cosine}


def compact_cosine(value):
    """Omit the leading zero and suppress rounded negative zero."""
    return f"{value:z.2f}".replace("0.", ".", 1) if abs(value) < 0.995 else f"{value:.2f}"


def render(
    output, metric, means, sds, columns=COLUMNS, positive_percent=None, anchored_style=False
):
    width = len(columns)
    boundaries = [i - 0.5 for i in range(1, width) if columns[i][1] != columns[i - 1][1]]
    edges = [-0.5, *boundaries, width - 0.5]
    groups = [
        ((a + b) / 2, {"joint": "Both", "pitch": "Pitch", "time": "Time"}[columns[int(a + 0.5)][1]])
        for a, b in zip(edges[:-1], edges[1:], strict=True)
    ]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    percentage = metric == "phrase-descent"
    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.5 if width == 7 else 4.9, 3.65))
    fig.subplots_adjust(left=0.245 if width == 7 else 0.18, right=0.985, top=0.87, bottom=0.17)
    from matplotlib import colormaps
    from matplotlib.colors import LinearSegmentedColormap

    turbo = colormaps["turbo_r"]
    anchored_cmap = LinearSegmentedColormap.from_list(
        "cosine_shifted_turbo",
        [
            (0.0, turbo(0.0)),
            (0.5, turbo(0.38)),
            (0.75, turbo(0.58)),
            (1.0, turbo(0.85)),
        ],
        N=1025,
    )
    im = ax.imshow(
        means,
        cmap=anchored_cmap if anchored_style else "cividis" if percentage else "RdBu",
        vmin=0 if percentage else -1,
        vmax=100 if percentage else 1,
        aspect="auto",
    )
    ax.set_yticks(range(11), LABELS)
    ax.set_xticks(range(width), [str(n) for n, _ in columns])
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=3)
    for center, title in groups:
        ax.text(center, -1.25, title, ha="center", va="bottom", clip_on=False)
    for i in range(11):
        for j in range(width):
            value = means[i, j]
            color = "white" if (value < 48 if percentage else abs(value) > 0.55) else "black"
            if anchored_style:
                rgb = np.array(im.cmap(im.norm(value))[:3])
                linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
                color = "black" if linear @ [0.2126, 0.7152, 0.0722] > 0.179 else "white"
            ax.text(
                j,
                i if percentage else i - 0.15,
                f"{value:.1f}"
                if percentage
                else compact_cosine(value)
                if positive_percent is not None or anchored_style
                else f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color=color,
            )
            if not percentage:
                ax.text(
                    j,
                    i + 0.23,
                    f"({positive_percent[i, j]:.1f}%)"
                    if positive_percent is not None
                    else f"±{compact_cosine(sds[i, j])}"
                    if anchored_style
                    else f"±{sds[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.2,
                    color=color,
                )
    ax.set_xticks(np.arange(-0.5, width), minor=True)
    ax.set_yticks(np.arange(-0.5, 11), minor=True)
    ax.grid(which="minor", color="white", alpha=0.4, linewidth=0.4)
    ax.tick_params(which="minor", length=0)
    for boundary in boundaries:
        ax.axvline(boundary, color="white", linewidth=1.1)
    for boundary in (1.5, 4.5, 7.5):
        ax.axhline(boundary, color="white", linewidth=1.1)
    cax = fig.add_axes([0.245 if width == 7 else 0.18, 0.085, 0.74 if width == 7 else 0.805, 0.025])
    cb = fig.colorbar(
        im,
        cax=cax,
        orientation="horizontal",
        ticks=[-1, -0.5, 0, 0.5, 1]
        if anchored_style
        else [0, 25, 50, 75, 100]
        if percentage
        else [-1, -0.5, 0, 0.5, 1],
    )
    cb.set_label(
        METRICS[metric] + ("; (positive phrases %)" if positive_percent is not None else ""),
        labelpad=2,
    )
    if anchored_style:
        cb.set_ticklabels(["−1", "−.5", "0", ".5", "1"])
    cb.ax.tick_params(length=2, pad=2)
    cb.ax.get_xticklabels()[0].set_horizontalalignment("left")
    cb.ax.get_xticklabels()[-1].set_horizontalalignment("right")
    stem = "phrase-cosine-positive" if positive_percent is not None else metric
    fig.savefig(output / f"{stem}.pdf")
    fig.savefig(output / f"{stem}.png", dpi=300)
    plt.close(fig)


def report(root, output, archive, extension=None):
    columns, registry = COLUMNS, targets()
    extra_counts = {}
    if extension is not None:
        from extend_gradient_cardinality import EXTENDED_COLUMNS, extra_targets

        columns = EXTENDED_COLUMNS
        registry += [(name, target) for name, target in extra_targets() if len(target) > 1]
        extra_sampling = json.loads((extension / "sampling.json").read_text())
        if not extra_sampling["complete"] or extra_sampling["signature"] != signature()[0]:
            raise ValueError("Incomplete or stale extension")
        extra_counts = extra_sampling["final_counts"]
    sampling = json.loads((root / "sampling.json").read_text())
    if not sampling["complete"] or sampling["signature"] != signature()[0]:
        raise ValueError("Incomplete or stale sampling")
    with (archive / "summary.csv").open() as source:
        old = {
            (int(r["events"]), r["condition"], r["loss"]): float(r["mean"])
            for r in csv.DictReader(source)
            if r["metric"] == "cosine"
        }
    output.mkdir(parents=True, exist_ok=True)
    rows, quality, hashes = [], [], {}
    matrices = {
        metric: (np.zeros((11, len(columns))), np.zeros((11, len(columns)))) for metric in METRICS
    }
    for column, (events, condition) in enumerate(columns):
        key = f"{events}-{condition}"
        source_root = extension if key in extra_counts else root
        count = (extra_counts if key in extra_counts else sampling["final_counts"])[key]
        collected = {metric: [] for metric in METRICS}
        for name, target in registry:
            if len(target) != events:
                continue
            path = shard_path(source_root, name, condition, count, 0)
            data = checked_data(path)
            np.testing.assert_array_equal(data["target"], target)
            metrics = alignment(data)
            for metric, values in metrics.items():
                collected[metric].append(values)
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            quality.append(
                dict(
                    target=name,
                    condition=condition,
                    candidates=count,
                    ties=int(data["ties"].sum()),
                    excluded=int(np.isnan(metrics["phrase-cosine"][:, 0]).sum()),
                )
            )
        for metric, blocks in collected.items():
            values = np.concatenate(blocks)
            for i, loss in enumerate(NAMES):
                valid = values[:, i][np.isfinite(values[:, i])]
                mean, sd = float(valid.mean()), float(valid.std(ddof=1))
                if metric == "event-cosine" and (events, condition, loss) in old:
                    np.testing.assert_allclose(mean, old[events, condition, loss], atol=1e-12)
                matrices[metric][0][i, column] = mean * (100 if metric == "phrase-descent" else 1)
                matrices[metric][1][i, column] = sd
                rows.append(
                    dict(
                        metric=metric,
                        events=events,
                        condition=condition,
                        loss=loss,
                        mean=mean,
                        candidate_sd=sd,
                        eligible=len(valid),
                        targets=len(blocks),
                        candidates_per_target=count,
                    )
                )
    write_csv(output / "summary.csv", rows)
    write_csv(output / "quality.csv", quality)
    for metric, (means, sds) in matrices.items():
        render(output, metric, means, sds, columns)
        write_csv(output / f"{metric}.csv", [r for r in rows if r["metric"] == metric])
        lines = [
            f"# {METRICS[metric]}",
            "",
            "| Loss | "
            + " | ".join(
                f"{ {'joint': 'Both', 'pitch': 'Pitch', 'time': 'Time'}[c] }: {n}"
                for n, c in columns
            )
            + " |",
            "|---|" + "---:|" * len(columns),
        ]
        for i, label in enumerate(TEXT_LABELS):
            cells = [
                f"{means[i, j]:.1f}"
                if metric == "phrase-descent"
                else f"{means[i, j]:.2f} ± {sds[i, j]:.2f}"
                for j in range(len(columns))
            ]
            lines.append("| " + " | ".join([label, *cells]) + " |")
        (output / f"{metric}.md").write_text("\n".join(lines) + "\n")
    save_json(
        output / "provenance.json",
        dict(
            signature=signature()[0],
            raw_artifacts=hashes,
            final_counts={**sampling["final_counts"], **extra_counts},
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            candidates=sum(q["candidates"] for q in quality),
            losses=NAMES,
            columns=columns,
            sd="Across candidate phrases, ddof=1; descriptive, not a confidence interval",
            validation=(
                "Archived event cosine means agree; single-event cosines agree; "
                "positive summed dot and phrase cosine agree"
            ),
        ),
    )
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/gradient-assessment-gpu"))
    parser.add_argument("--output", type=Path, default=Path("docs/gradient-assessment/alignment"))
    parser.add_argument("--archive", type=Path, default=Path("docs/gradient-assessment"))
    parser.add_argument("--extension", type=Path)
    args = parser.parse_args()
    report(args.root, args.output, args.archive, args.extension)
