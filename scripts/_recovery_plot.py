"""Presentation of completed recovery results; no synthesis on import."""

import hashlib
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from icassp27_phrase.synth.config import CARDINALITIES, MASTER_SEED

recovery = SimpleNamespace(CARDINALITIES=CARDINALITIES, TARGETS_PER_CELL=150)
REPORT_LOSSES = (
    "single_stft",
    "smooth_mss",
    "sot_published_composite",
    "linear_jtfot",
    "log_jtfot",
    "cel",
    "log_cel",
    "dec_cel",
    "tlog_cel",
)


REPORT_LABELS = (
    "SS",
    "SmoMSS",
    "SOT",
    "TFW2",
    "logTFW2",
    "CeL",
    "logCeL",
    "decCeL",
    "tlogCeL",
)


LSD_FIGURE_LOSSES = ("single_stft", "log_jtfot", "dec_cel", "random")


LSD_FIGURE_LABELS = (
    "SS",
    r"log$\mathrm{TF}\mathcal{W}_2$",
    r"decCe$\mathcal{L}$",
    "Random",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
        os.replace(temporary, path)
        os.chmod(path, 0o644)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def random_derangement(cardinality: int) -> np.ndarray:
    """Return a fixed no-self-match permutation for the random LSD baseline."""
    size = recovery.TARGETS_PER_CELL
    original = np.arange(size)
    generator = np.random.default_rng(MASTER_SEED + cardinality)
    while True:
        permutation = generator.permutation(size)
        if np.all(permutation != original):
            return permutation


def format_value(value: float) -> str:
    if value < 0.01:
        return r"$<$.01"
    rendered = f"{value:.3g}"
    if "e" in rendered:
        mantissa, exponent = rendered.split("e", 1)
        rendered = f"{mantissa}e{int(exponent)}"
    return rendered


def format_ranked_value(value: float, rank: int) -> str:
    rendered = format_value(value)
    if rank == 1:
        return rf"\textbf{{{rendered}}}"
    if rank == 2:
        return rf"\underline{{{rendered}}}"
    if rank == 3:
        return rf"\textit{{{rendered}}}"
    return rendered


def render_table(medians: dict, path: Path) -> None:
    labels = (
        "SS",
        "SmoMSS",
        "SOT",
        r"$\mathrm{TF}\mathcal{W}_2$",
        r"log$\mathrm{TF}\mathcal{W}_2$",
        r"Ce$\mathcal L$",
        r"logCe$\mathcal L$",
        r"decCe$\mathcal L$",
        r"tlogCe$\mathcal L$",
    )
    configurations = (
        ("--", "--", "--"),
        ("--", "--", "--"),
        ("--", "--", "--"),
        ("--", "--", "--"),
        ("--", "--", "--"),
        (r"$\fourdiagonals$", "Lin.", r"$\times$"),
        (r"$\fourdiagonals$", "Log", r"$\times$"),
        (r"$\fourdiagonals$", "Lin.", r"$\checkmark$"),
        (r"$\forwarddiagonals$", "Log", r"$\times$"),
    )
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\newcommand{\fourdiagonals}{\mathord{\ooalign{%",
        r"  \hfil$\nearrow$\hfil\cr",
        r"  \hfil$\nwarrow$\hfil\cr",
        r"  \hfil$\searrow$\hfil\cr",
        r"  \hfil$\swarrow$\hfil\cr}}}",
        r"\newcommand{\forwarddiagonals}{\mathord{\ooalign{%",
        r"  \hfil$\nearrow$\hfil\cr",
        r"  \hfil$\searrow$\hfil\cr}}}",
        (
            r"\begin{tabularx}{\linewidth}{@{}lccc*{5}{>{\centering\arraybackslash}X}|"
            r"*{5}{>{\centering\arraybackslash}X}@{}}"
        ),
        r"& & &",
        (
            r"& \multicolumn{5}{c|}{$\Delta f_0$ (cents)} & "
            r"\multicolumn{5}{c}{$\Delta t$ (ms)} \\"
        ),
        r"Loss & Directions & Weighting & Decay",
        (
            r"& 1 note & 2 notes & 4 notes & 6 notes & 8 notes "
            r"& 1 note & 2 notes & 4 notes & 6 notes & 8 notes \\"
        ),
        r"\midrule",
    ]
    for index, (loss, label, configuration) in enumerate(
        zip(REPORT_LOSSES, labels, configurations, strict=True)
    ):
        cells = []
        for metric in ("pitch_mae_cents", "onset_mae_ms"):
            for cardinality in recovery.CARDINALITIES:
                values = {
                    candidate: medians[metric][candidate, cardinality]
                    for candidate in REPORT_LOSSES
                }
                # Displayed values below .01 share the best rank. Other exact
                # ties also share a rank, so the next distinct value follows.
                groups = sorted({0.0 if value < 0.01 else value for value in values.values()})
                value = values[loss]
                group = 0.0 if value < 0.01 else value
                cells.append(format_ranked_value(value, groups.index(group) + 1))
        lines.append(" & ".join((label, *configuration, *cells)) + r" \\")
        if index == 4:
            lines.append(r"\midrule")
    lines.extend((r"\bottomrule", r"\end{tabularx}", r"\endgroup"))
    atomic_text(path, "\n".join(lines) + "\n")


def save_figure(figure, path: Path, *, pad_inches: float = 0.1) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    try:
        metadata = {"CreationDate": None, "ModDate": None} if path.suffix == ".pdf" else None
        figure.savefig(
            temporary,
            dpi=300,
            bbox_inches="tight",
            pad_inches=pad_inches,
            metadata=metadata,
        )
        os.replace(temporary, path)
        os.chmod(path, 0o644)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def render_lsd(lsd: dict, output_stem: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch

    plt.rcdefaults()
    cmap = plt.get_cmap("magma")
    colors = {
        "single_stft": cmap(0.12),
        "log_jtfot": cmap(0.43),
        "dec_cel": cmap(0.74),
        "random": (0.52, 0.52, 0.52, 1.0),
    }
    figure, axis = plt.subplots(figsize=(3.5, 1.9))
    offsets = np.linspace(-0.27, 0.27, len(LSD_FIGURE_LOSSES))
    for loss_index, loss in enumerate(LSD_FIGURE_LOSSES):
        for cardinality_index, cardinality in enumerate(recovery.CARDINALITIES):
            values = np.asarray(
                [lsd[loss, cardinality, index] for index in range(recovery.TARGETS_PER_CELL)]
            )
            position = cardinality_index + offsets[loss_index]
            violin = axis.violinplot(
                values,
                positions=[position],
                widths=0.17,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in violin["bodies"]:
                body.set_facecolor(colors[loss])
                body.set_edgecolor("black")
                body.set_linewidth(0.35)
                body.set_alpha(0.68)
            lower, median, upper = np.quantile(values, (0.25, 0.5, 0.75))
            axis.vlines(position, lower, upper, color="white", linewidth=0.85)
            axis.scatter(
                position,
                median,
                s=8,
                color="white",
                edgecolor="black",
                linewidth=0.35,
                zorder=4,
            )
    axis.set_xticks(range(len(recovery.CARDINALITIES)), recovery.CARDINALITIES)
    axis.set_xlabel("Number of notes", fontsize=8, labelpad=5)
    axis.set_ylabel("LSD (dB)", fontsize=8)
    axis.tick_params(labelsize=7, length=2.0, pad=1.2)
    axis.grid(axis="y", color="0.88", linewidth=0.45)
    handles = [
        Patch(facecolor=colors[loss], edgecolor="none", label=label)
        for loss, label in zip(LSD_FIGURE_LOSSES, LSD_FIGURE_LABELS, strict=True)
    ]
    axis.legend(
        handles=handles,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=5.7,
        handlelength=1.05,
        handleheight=1.05,
        handletextpad=0.35,
        labelspacing=0.5,
        borderaxespad=0.0,
    )
    for spine in axis.spines.values():
        spine.set_linewidth(0.6)
    figure.subplots_adjust(left=0.16, right=0.74, bottom=0.26, top=0.98)
    paths = {suffix: output_stem.with_suffix(f".{suffix}") for suffix in ("pdf", "png")}
    for path in paths.values():
        save_figure(figure, path, pad_inches=0.08)
    plt.close(figure)
    return {suffix: sha256(path) for suffix, path in paths.items()}
