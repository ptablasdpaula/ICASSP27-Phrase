"""The submitted mean ± SD gradient heatmap, without exploratory style variants."""

import numpy as np


def compact_cosine(value):
    """Omit the leading zero and suppress rounded negative zero."""
    return f"{value:z.2f}".replace("0.", ".", 1) if abs(value) < 0.995 else f"{value:.2f}"


def render(output, means, sds, columns, labels, output_stem="phrase-cosine"):
    if sds.shape != means.shape or len(labels) != means.shape[0]:
        raise ValueError("Means, SDs and row labels must have matching dimensions")
    width = len(columns)
    if width != means.shape[1] or not np.isfinite(means).all() or not np.isfinite(sds).all():
        raise ValueError("Incomplete gradient summary")
    boundaries = [i - 0.5 for i in range(1, width) if columns[i][1] != columns[i - 1][1]]
    edges = [-0.5, *boundaries, width - 0.5]
    titles = {
        "joint": "Joint",
        "pitch": "Known time",
        "time": r"Known $f_0$",
        "persistent": "State",
        "controls": "7-D",
    }
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(4.35, 3.45))
    fig.subplots_adjust(left=0.105, right=0.995, top=0.905, bottom=0.145)
    im = ax.imshow(means, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(len(labels)), labels)
    for label in ax.get_yticklabels():
        label.set_rotation(-18)
        label.set_ha("right")
        label.set_rotation_mode("anchor")
    ax.set_xticks(range(width), [str(n) for n, _ in columns])
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=3)
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        ax.text(
            (a + b) / 2,
            -1.25,
            titles[columns[int(a + 0.5)][1]],
            ha="center",
            va="bottom",
            clip_on=False,
        )
    for i in range(len(labels)):
        for j in range(width):
            value = means[i, j]
            rgb = np.array(im.cmap(im.norm(value))[:3])
            linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
            colour = "black" if linear @ [0.2126, 0.7152, 0.0722] > 0.179 else "white"
            ax.text(
                j,
                i - 0.15,
                compact_cosine(value),
                ha="center",
                va="center",
                fontsize=6.5,
                color=colour,
            )
            ax.text(
                j,
                i + 0.23,
                f"±{compact_cosine(sds[i, j])}",
                ha="center",
                va="center",
                fontsize=5.2,
                color=colour,
            )
    ax.set_xticks(np.arange(-0.5, width), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels)), minor=True)
    ax.grid(which="minor", color="white", alpha=0.4, linewidth=0.4)
    ax.tick_params(which="minor", length=0)
    for boundary in boundaries:
        ax.axvline(boundary, color="white", linewidth=1.1)
    for boundary in (1.5, 4.5, 7.5):
        ax.axhline(boundary, color="white", linewidth=1.1)
    cax = fig.add_axes([0.105, 0.112, 0.89, 0.02])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 0.5, 1])
    bar.set_label("Mean whole-phrase cosine", labelpad=2)
    bar.ax.set_xlim(-0.1, 1)
    bar.set_ticklabels(["0", ".5", "1"])
    bar.ax.tick_params(length=2, pad=2)
    bar.ax.get_xticklabels()[0].set_horizontalalignment("left")
    bar.ax.get_xticklabels()[-1].set_horizontalalignment("right")
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{output_stem}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(output / f"{output_stem}.png", dpi=300)
    plt.close(fig)
