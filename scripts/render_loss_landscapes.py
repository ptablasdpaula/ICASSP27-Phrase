#!/usr/bin/env python3
"""Maintain the paper's single-event loss-gradient diagnostic.

The qualified Fourier--Thiran panels are recovered losslessly from the existing
Matplotlib PDF rather than recomputed. Only the Naive--Linear sensitivity
comparison invokes the recursive renderer, and that command requires CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from icassp27_phrase.config import ExciterConfig, WaveguideConfig
from icassp27_phrase.losses import build_loss
from icassp27_phrase.runtime import configure_reproducibility, require_accelerated_backend
from icassp27_phrase.synth import PhraseSynth
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIRECTORY = ROOT / "paper" / "figures"
DEFAULT_OUTPUT_STEM = FIGURE_DIRECTORY / "loss_landscapes"
DEFAULT_QUALIFIED_CACHE = FIGURE_DIRECTORY / "loss_landscapes.fourier_thiran.npz"
DEFAULT_COMPARISON_CACHE = FIGURE_DIRECTORY / "loss_landscapes.naive_linear.npz"
TARGET_F0_HZ = 160.0
TARGET_ONSET_SECONDS = 1.0
SURFACE_POINTS = 25
DISPLAY_INDICES = tuple(range(2, 23, 2))
TARGET_INDEX = 12
BASE_LOSS_NAMES = (
    "waveform_l1",
    "waveform_mse",
    "smooth_mss",
    "sot_published_composite",
    "linear_jtfot",
    "bidirectional_cumulative_energy",
)
PAPER_LOSS_LABELS = (
    r"$L_1$",
    r"$L_2$",
    "MSS",
    "SOT",
    r"$\mathrm{TF}\mathcal{W}_2$",
    r"$\mathrm{BiCu}\mathcal{L}$",
)
EXPECTED_FOURIER_THIRAN_COUNTS = np.asarray((72, 59, 51, 62, 91, 120))
QUALIFIED_CACHE_SCHEMA = "qualified-fourier-thiran-landscape-cache-v1"
COMPARISON_CACHE_SCHEMA = "cuda-naive-linear-gradient-cache-v1"


@dataclass(frozen=True)
class RendererSpec:
    """One renderer used by the diagnostic."""

    name: str
    display_name: str
    exciter: ExciterConfig
    waveguide: WaveguideConfig


@dataclass(frozen=True)
class LandscapeFigureStyle:
    """Registered single-column landscape styling."""

    arrow_color: str = "white"
    annotation_color: str = "white"
    arrow_half_length: float = 0.026
    arrow_mutation_scale: float = 4.2
    arrow_linewidth: float = 0.48
    annotation_x: float = 0.965
    annotation_y: float = 0.985
    plot_left: float = 0.13
    plot_right: float = 0.985
    plot_bottom: float = 0.12
    plot_top: float = 0.985
    horizontal_space: float = 0.065
    vertical_space: float = 0.12
    colorbar_bottom: float = 0.035
    colorbar_height: float = 0.016

    @property
    def horizontal_center(self) -> float:
        """Centre shared labels on the axes/colorbar span, not the canvas."""
        return 0.5 * (self.plot_left + self.plot_right)


LANDSCAPE_FIGURE_STYLE = LandscapeFigureStyle()


def renderer_specs() -> tuple[RendererSpec, RendererSpec]:
    """Return the campaign renderer and the simple sensitivity comparison."""
    return (
        RendererSpec(
            name="fourier_thiran",
            display_name="Fourier--Thiran",
            exciter=ExciterConfig(method="fourier", fourier_fft_length=262_144),
            waveguide=WaveguideConfig(
                interpolation="thiran", realization="df2", state_policy="hard_reset"
            ),
        ),
        RendererSpec(
            name="naive_linear",
            display_name="Naive--Linear",
            exciter=ExciterConfig(method="naive"),
            waveguide=WaveguideConfig(
                interpolation="linear", realization="df2", state_policy="hard_reset"
            ),
        ),
    )


def coordinate_grids() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pitch/onset unit axes and flattened pitch-onset coordinates."""
    pitch_unit = np.linspace(0.0, 1.0, SURFACE_POINTS, dtype=np.float64)
    onset_unit = np.linspace(0.0, 1.0, SURFACE_POINTS, dtype=np.float64)
    pitch_unit[TARGET_INDEX] = 0.5
    onset_unit[TARGET_INDEX] = 0.5
    pitch_mesh, onset_mesh = np.meshgrid(pitch_unit, onset_unit, indexing="ij")
    coordinates = np.stack((pitch_mesh.reshape(-1), onset_mesh.reshape(-1)), axis=-1)
    if (
        coordinates.shape != (SURFACE_POINTS**2, 2)
        or np.count_nonzero(pitch_unit == 0.5) != 1
        or np.count_nonzero(onset_unit == 0.5) != 1
        or not np.all(np.diff(pitch_unit) > 0.0)
        or not np.all(np.diff(onset_unit) > 0.0)
    ):
        raise RuntimeError("the target-inclusive landscape grid is malformed")
    return pitch_unit, onset_unit, coordinates


def displayed_coordinates() -> np.ndarray:
    """Return the 120 displayed non-target points in pitch-onset order."""
    pitch_unit, onset_unit, _ = coordinate_grids()
    points = [
        (pitch_unit[row], onset_unit[column])
        for row in DISPLAY_INDICES
        for column in DISPLAY_INDICES
        if (row, column) != (TARGET_INDEX, TARGET_INDEX)
    ]
    result = np.asarray(points, dtype=np.float64)
    if result.shape != (120, 2):
        raise RuntimeError("the display sublattice must contain 120 non-target points")
    return result


def controls(coordinates: Tensor) -> tuple[Tensor, Tensor]:
    """Map normalized pitch-onset coordinates to registered physical controls."""
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape [batch,2]")
    f0_hz = 80.0 * torch.pow(4.0, coordinates[:, 0:1])
    onset_seconds = 0.2 + 1.6 * coordinates[:, 1:2]
    return f0_hz, onset_seconds


def target_direction_summary(coordinates: np.ndarray, descent: np.ndarray) -> dict[str, Any]:
    """Count descent vectors forming an acute angle with target displacement."""
    positions = np.asarray(coordinates, dtype=np.float64)
    directions = np.asarray(descent, dtype=np.float64)
    if positions.shape != (120, 2) or directions.shape != positions.shape:
        raise ValueError("direction summaries require the 120 non-target points")
    if not np.isfinite(positions).all() or not np.isfinite(directions).all():
        raise FloatingPointError("direction summary contains non-finite values")
    dot = np.sum(directions * (0.5 - positions), axis=1)
    count = int(np.count_nonzero(dot > 0.0))
    return {
        "target_directed_count": count,
        "displayed_non_target_gradients": 120,
        "target_directed_percentage": round(100.0 * count / 120.0, 1),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha256(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    _atomic_bytes(path, buffer.getvalue())


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    signed = dict(payload)
    signed["payload_sha256"] = _payload_sha256(payload)
    value = json.dumps(signed, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_bytes(path, value.encode())


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _pdf_object(document: bytes, number: int) -> bytes:
    marker = re.search(rb"(?m)^" + str(number).encode() + rb" 0 obj\r?\n", document)
    if marker is None:
        raise ValueError(f"PDF object {number} is missing")
    return document[marker.end() : document.index(b"endobj", marker.end())]


def _pdf_stream(document: bytes, body: bytes) -> bytes:
    marker = body.index(b"stream") + len(b"stream")
    if body[marker : marker + 2] == b"\r\n":
        marker += 2
    elif body[marker : marker + 1] in (b"\r", b"\n"):
        marker += 1
    header = body[:marker]
    indirect = re.search(rb"/Length (\d+) 0 R", header)
    if indirect is not None:
        length = int(_pdf_object(document, int(indirect.group(1))).strip())
    else:
        direct = re.search(rb"/Length (\d+)", header)
        if direct is None:
            raise ValueError("PDF stream has no declared length")
        length = int(direct.group(1))
    compressed = body[marker : marker + length]
    return zlib.decompress(compressed) if b"/FlateDecode" in header else compressed


def _pdf_literal(data: bytes, start: int) -> tuple[bytes, int]:
    """Decode one parenthesized PDF byte string."""
    if data[start : start + 1] != b"(":
        raise ValueError("PDF literal does not start with an opening parenthesis")
    result = bytearray()
    depth = 1
    index = start + 1
    simple_escapes = {
        ord("n"): ord("\n"),
        ord("r"): ord("\r"),
        ord("t"): ord("\t"),
        ord("b"): ord("\b"),
        ord("f"): ord("\f"),
    }
    while depth:
        value = data[index]
        index += 1
        if value == ord("\\"):
            escaped = data[index]
            index += 1
            if escaped in simple_escapes:
                result.append(simple_escapes[escaped])
            elif escaped in (ord("("), ord(")"), ord("\\")):
                result.append(escaped)
            elif ord("0") <= escaped <= ord("7"):
                digits = bytearray((escaped,))
                for _ in range(2):
                    if index < len(data) and ord("0") <= data[index] <= ord("7"):
                        digits.append(data[index])
                        index += 1
                    else:
                        break
                result.append(int(digits.decode(), 8))
            elif escaped == ord("\r"):
                if index < len(data) and data[index] == ord("\n"):
                    index += 1
            elif escaped != ord("\n"):
                result.append(escaped)
        elif value == ord("("):
            depth += 1
            result.append(value)
        elif value == ord(")"):
            depth -= 1
            if depth:
                result.append(value)
        else:
            result.append(value)
    return bytes(result), index


def _decode_indexed_image(document: bytes, object_number: int) -> np.ndarray:
    body = _pdf_object(document, object_number)
    header = body[: body.index(b"stream")]
    width_match = re.search(rb"/Width (\d+)", header)
    height_match = re.search(rb"/Height (\d+)", header)
    palette_match = re.search(rb"/Indexed /DeviceRGB (\d+)\s*", header)
    if width_match is None or height_match is None or palette_match is None:
        raise ValueError("qualified PDF does not contain indexed Matplotlib images")
    width = int(width_match.group(1))
    height = int(height_match.group(1))
    maximum_index = int(palette_match.group(1))
    literal_start = header.index(b"(", palette_match.end())
    palette_bytes, _ = _pdf_literal(header, literal_start)
    if len(palette_bytes) != 3 * (maximum_index + 1):
        raise ValueError("qualified PDF has a malformed indexed-color palette")
    palette = np.frombuffer(palette_bytes, dtype=np.uint8).reshape(-1, 3)
    filtered = np.frombuffer(_pdf_stream(document, body), dtype=np.uint8)
    rows = filtered.reshape(height, width + 1)
    if not np.all(rows[:, 0] == 0):
        raise ValueError("qualified PDF image does not use the PNG-none predictor")
    return palette[rows[:, 1:]].copy()


def extract_qualified_cache(source_pdf: Path, output: Path) -> dict[str, Any]:
    """Recover the already-computed surfaces and arrows from the current PDF."""
    document = source_pdf.read_bytes()
    content_reference = re.search(rb"/Contents (\d+) 0 R", document)
    if content_reference is None:
        raise ValueError("qualified PDF has no page content stream")
    content = _pdf_stream(document, _pdf_object(document, int(content_reference.group(1))))

    xobject_body: bytes | None = None
    object_pattern = re.compile(rb"(?m)^(\d+) 0 obj\r?\n")
    for match in object_pattern.finditer(document):
        body = document[match.end() : document.index(b"endobj", match.end())]
        if all(f"/I{index} ".encode() in body for index in range(1, 7)):
            xobject_body = body
            break
    if xobject_body is None:
        raise ValueError("qualified PDF has no six-panel image dictionary")
    image_objects = []
    for index in range(1, 7):
        match = re.search(rb"/I" + str(index).encode() + rb" (\d+) 0 R", xobject_body)
        if match is None:
            raise ValueError(f"qualified PDF image I{index} is missing")
        image_objects.append(int(match.group(1)))
    surfaces_rgb = np.stack(
        [_decode_indexed_image(document, number) for number in image_objects], axis=0
    )

    arrow_blocks = re.findall(
        rb"/A3\s+gs 1 g\s+1 j 0 G 1 g\s+(.*?)\s+f", content, re.DOTALL
    )
    if len(arrow_blocks) != 6 * 120:
        raise ValueError("qualified PDF does not contain exactly 720 white arrows")
    direction_rows: list[np.ndarray] = []
    vertex_pattern = re.compile(rb"(?m)^(-?[0-9.]+) (-?[0-9.]+) [ml]$")
    for block in arrow_blocks:
        vertices = np.asarray(
            [(float(x), float(y)) for x, y in vertex_pattern.findall(block)],
            dtype=np.float64,
        )
        if vertices.shape != (7, 2):
            raise ValueError("qualified PDF contains a malformed arrow path")
        direction_xy = vertices[0] - vertices[3:5].mean(axis=0)
        direction_xy /= np.linalg.norm(direction_xy)
        direction_rows.append(direction_xy[::-1])  # PDF x/y -> pitch/onset.
    directions = np.stack(direction_rows).reshape(6, 120, 2)
    positions = displayed_coordinates()
    counts = np.asarray(
        [
            target_direction_summary(positions, value)["target_directed_count"]
            for value in directions
        ]
    )
    if not np.array_equal(counts, EXPECTED_FOURIER_THIRAN_COUNTS):
        raise RuntimeError(
            "recovered Fourier--Thiran arrows do not reproduce the paper counts: "
            f"{counts.tolist()}"
        )
    _atomic_npz(
        output,
        schema=np.asarray(QUALIFIED_CACHE_SCHEMA),
        surfaces_rgb=surfaces_rgb,
        directions=directions,
        counts=counts,
        source_pdf_sha256=np.asarray(_sha256(source_pdf)),
    )
    return {
        "cache": str(output),
        "cache_sha256": _sha256(output),
        "source_pdf_sha256": _sha256(source_pdf),
        "counts": counts.tolist(),
    }


def _target_audio(synth: PhraseSynth, device: torch.device) -> Tensor:
    coordinates = torch.tensor([[0.5, 0.5]], dtype=torch.float64, device=device)
    with torch.no_grad():
        f0_hz, onset_seconds = controls(coordinates)
        return synth(f0_hz, onset_seconds)[0].detach()


def compute_directions(
    synth: PhraseSynth,
    target_audio: Tensor,
    *,
    device: torch.device,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute six losses from each shared CUDA renderer graph chunk."""
    positions = displayed_coordinates()
    metrics = {name: build_loss(name, target_audio) for name in BASE_LOSS_NAMES}
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in BASE_LOSS_NAMES}
    for begin in range(0, positions.shape[0], chunk_size):
        coordinates = torch.tensor(
            positions[begin : begin + chunk_size],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        f0_hz, onset_seconds = controls(coordinates)
        candidate_audio = synth(f0_hz, onset_seconds)
        if not bool(torch.isfinite(candidate_audio.detach()).all()):
            raise FloatingPointError("the comparison renderer produced non-finite audio")
        for loss_index, loss_name in enumerate(BASE_LOSS_NAMES):
            values = metrics[loss_name].distances(candidate_audio)
            if not bool(torch.isfinite(values.detach()).all()):
                raise FloatingPointError(f"{loss_name} comparison loss is non-finite")
            gradient, = torch.autograd.grad(
                values.sum(),
                coordinates,
                retain_graph=loss_index < len(BASE_LOSS_NAMES) - 1,
            )
            chunks[loss_name].append(-gradient.detach().cpu().numpy())
    directions = []
    counts = []
    for loss_name in BASE_LOSS_NAMES:
        raw = np.concatenate(chunks[loss_name], axis=0)
        norm = np.linalg.norm(raw, axis=1, keepdims=True)
        unit = np.divide(raw, norm, out=np.zeros_like(raw), where=norm > 0.0)
        directions.append(unit)
        counts.append(target_direction_summary(positions, unit)["target_directed_count"])
    return np.stack(directions), np.asarray(counts)


def compute_comparison_cache(device_name: str, chunk_size: int, output: Path) -> dict[str, Any]:
    """Run only the new Naive--Linear direction diagnostic on a GPU."""
    if chunk_size < 1:
        raise ValueError("gradient chunk size must be positive")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError("recursive landscape diagnostics must run on CUDA")
    configure_reproducibility()
    require_accelerated_backend(device)
    comparison = renderer_specs()[1]
    synth = PhraseSynth(
        exciter_config=comparison.exciter, waveguide_config=comparison.waveguide
    ).to(device)
    directions, counts = compute_directions(
        synth,
        _target_audio(synth, device),
        device=device,
        chunk_size=chunk_size,
    )
    if directions.shape != (6, 120, 2) or np.any(counts < 0) or np.any(counts > 120):
        raise RuntimeError("Naive--Linear diagnostic produced malformed results")
    _atomic_npz(
        output,
        schema=np.asarray(COMPARISON_CACHE_SCHEMA),
        directions=directions,
        counts=counts,
        device=np.asarray(str(device)),
        renderer_json=np.asarray(json.dumps(synth.provenance(), sort_keys=True)),
        source_commit=np.asarray(_git_commit()),
        torch_version=np.asarray(torch.__version__),
    )
    return {
        "cache": str(output),
        "cache_sha256": _sha256(output),
        "counts": counts.tolist(),
        "percentages": [round(100.0 * int(value) / 120.0, 1) for value in counts],
    }


def _load_cache(path: Path, schema: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name].copy() for name in archive.files}
    if str(values.get("schema", "")) != schema:
        raise ValueError(f"{path} does not use schema {schema!r}")
    return values


def _diagnostics(counts: np.ndarray) -> dict[str, dict[str, Any]]:
    if counts.shape != (6,):
        raise ValueError("a renderer cache must contain six direction counts")
    return {
        name: {
            "target_directed_count": int(count),
            "displayed_non_target_gradients": 120,
            "target_directed_percentage": round(100.0 * int(count) / 120.0, 1),
        }
        for name, count in zip(BASE_LOSS_NAMES, counts, strict=True)
    }


def _atomic_figure(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    try:
        metadata = (
            {
                "Title": "Single-event loss-gradient diagnostic",
                "Subject": "Fourier onset placement and Thiran fractional delay",
                "Creator": "scripts/render_loss_landscapes.py",
                "CreationDate": None,
                "ModDate": None,
            }
            if path.suffix == ".pdf"
            else None
        )
        figure.savefig(
            temporary,
            dpi=300,
            bbox_inches="tight",
            metadata=metadata,
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def render_figure(
    surfaces_rgb: np.ndarray,
    directions: np.ndarray,
    counts: np.ndarray,
    output: Path,
    labels: tuple[str, ...] = PAPER_LOSS_LABELS,
) -> None:
    """Restyle the already-computed Fourier--Thiran six-panel figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    if surfaces_rgb.shape != (6, 180, 180, 3) or directions.shape != (6, 120, 2):
        raise ValueError("qualified landscape cache has unexpected array shapes")
    positions = displayed_coordinates()
    style = LANDSCAPE_FIGURE_STYLE
    figure, axes = plt.subplots(3, 2, figsize=(4.65, 7.15), sharex=True, sharey=True)
    cmap = plt.get_cmap("magma")
    for panel, (axis, _loss_name, label) in enumerate(
        zip(axes.reshape(-1), BASE_LOSS_NAMES, labels, strict=True)
    ):
        axis.imshow(
            surfaces_rgb[panel],
            origin="upper",
            extent=(-0.8, 0.8, -1.0, 1.0),
            interpolation="nearest",
            aspect="auto",
        )
        for coordinate, direction in zip(positions, directions[panel], strict=True):
            if np.linalg.norm(direction) == 0.0:
                continue
            centre = np.asarray([coordinate[1], coordinate[0]])
            display_direction = np.asarray([direction[1], direction[0]])
            arrow = FancyArrowPatch(
                centre - style.arrow_half_length * display_direction,
                centre + style.arrow_half_length * display_direction,
                transform=axis.transAxes,
                arrowstyle="-|>",
                mutation_scale=style.arrow_mutation_scale,
                linewidth=style.arrow_linewidth,
                color=style.arrow_color,
                shrinkA=0.0,
                shrinkB=0.0,
                clip_on=True,
                zorder=4,
            )
            axis.add_patch(arrow)
        axis.plot(
            0.0,
            0.0,
            marker="+",
            markersize=10.5,
            markeredgewidth=1.8,
            color="#fff59d",
            zorder=5,
        )
        percentage = 100.0 * int(counts[panel]) / 120.0
        axis.text(
            style.annotation_x,
            style.annotation_y,
            f"{percentage:.1f}%",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7.2,
            color=style.annotation_color,
            zorder=6,
        )
        axis.set_title(label, fontsize=11, pad=2.0)
        axis.set_box_aspect(1)
        axis.set_xticks((-0.8, 0.0, 0.8), (r"$-0.8$", "0", r"$+0.8$"))
        x_tick_labels = axis.get_xticklabels()
        if x_tick_labels:
            x_tick_labels[0].set_horizontalalignment("left")
            x_tick_labels[-1].set_horizontalalignment("right")
        axis.set_yticks((-1.0, 0.0, 1.0), (r"$-1$", "0", r"$+1$"))
        axis.tick_params(labelsize=8, length=2.2, pad=1.0)
        for spine in axis.spines.values():
            spine.set_linewidth(0.65)
    figure.subplots_adjust(
        left=style.plot_left,
        right=style.plot_right,
        bottom=style.plot_bottom,
        top=style.plot_top,
        hspace=style.vertical_space,
        wspace=style.horizontal_space,
    )
    figure.supxlabel(
        "Time shift (s)", fontsize=10, x=style.horizontal_center, y=0.065
    )
    mss_position = axes[1, 0].get_position()
    figure.supylabel(
        r"$f_0$ shift (octaves)",
        fontsize=10,
        x=0.035,
        y=0.5 * (mss_position.y0 + mss_position.y1),
    )
    color_axis = figure.add_axes(
        (
            style.plot_left,
            style.colorbar_bottom,
            style.plot_right - style.plot_left,
            style.colorbar_height,
        )
    )
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(0.0, 1.0), cmap=cmap),
        cax=color_axis,
        orientation="horizontal",
        ticks=(0.0, 0.5, 1.0),
    )
    colorbar.ax.xaxis.set_ticks_position("bottom")
    colorbar.ax.tick_params(
        axis="x",
        which="both",
        labelsize=7,
        length=2,
        pad=1,
        labeltop=False,
        labelbottom=True,
    )
    color_tick_labels = colorbar.ax.get_xticklabels()
    color_tick_labels[0].set_horizontalalignment("left")
    color_tick_labels[-1].set_horizontalalignment("right")
    colorbar.set_label("Loss (normalised)", fontsize=8, labelpad=1)
    _atomic_figure(figure, output)
    plt.close(figure)


def build_figure(
    qualified_cache: Path,
    comparison_cache: Path,
    output_stem: Path,
) -> dict[str, Any]:
    qualified = _load_cache(qualified_cache, QUALIFIED_CACHE_SCHEMA)
    comparison = _load_cache(comparison_cache, COMPARISON_CACHE_SCHEMA)
    fourier_counts = qualified["counts"].astype(np.int64)
    comparison_counts = comparison["counts"].astype(np.int64)
    if not np.array_equal(fourier_counts, EXPECTED_FOURIER_THIRAN_COUNTS):
        raise RuntimeError("qualified Fourier--Thiran counts changed")
    if comparison["directions"].shape != (6, 120, 2):
        raise ValueError("Naive--Linear cache has malformed directions")
    output_stem = output_stem.resolve()
    figure_path = output_stem.with_suffix(".pdf")
    render_figure(
        qualified["surfaces_rgb"],
        qualified["directions"],
        fourier_counts,
        figure_path,
    )
    campaign, comparison_spec = renderer_specs()
    diagnostics = {
        campaign.name: _diagnostics(fourier_counts),
        comparison_spec.name: _diagnostics(comparison_counts),
    }
    renderer_payload = {
        spec.name: {
            "display_name": spec.display_name,
            "exciter": asdict(spec.exciter),
            "waveguide": asdict(spec.waveguide),
        }
        for spec in (campaign, comparison_spec)
    }
    payload = {
        "schema": "single-event-two-renderer-gradient-diagnostic-v1",
        "source_commit": _git_commit(),
        "target": {
            "f0_hz": TARGET_F0_HZ,
            "onset_seconds": TARGET_ONSET_SECONDS,
        },
        "surface_grid": {
            "shape": [SURFACE_POINTS, SURFACE_POINTS],
            "f0_hz": [80.0, 320.0],
            "onset_seconds": [0.2, 1.8],
            "pitch_coordinate": "log2(f0/80)/2",
            "onset_coordinate": "(onset-0.2)/1.6",
        },
        "direction_metric": {
            "display_indices": list(DISPLAY_INDICES),
            "target_index": TARGET_INDEX,
            "criterion": "dot(-unit_gradient, target_minus_candidate) > 0",
            "displayed_non_target_gradients": 120,
        },
        "losses": list(BASE_LOSS_NAMES),
        "renderers": renderer_payload,
        "diagnostics": diagnostics,
        "reuse": {
            "fourier_thiran": (
                "surfaces and arrow directions recovered from the existing qualified PDF; "
                "no recursive rerender"
            ),
            "qualified_source_pdf_sha256": str(qualified["source_pdf_sha256"]),
            "qualified_cache_sha256": _sha256(qualified_cache),
            "naive_linear": "new direction-only comparison computed on CUDA",
            "comparison_device": str(comparison["device"]),
            "comparison_source_commit": str(comparison["source_commit"]),
            "comparison_torch_version": str(comparison["torch_version"]),
            "comparison_cache_sha256": _sha256(comparison_cache),
        },
        "figure": {
            "renderer": campaign.name,
            "layout": "single-column 3x2 independently normalized loss surfaces",
            "arrows": (
                "pure white, no outline; half-length 0.026 axes units, "
                "mutation scale 4.2, linewidth 0.48"
            ),
            "panel_annotation": "pure-white target-directed percentage at upper right",
            "x_label_alignment": "horizontal center of the plot and colorbar span",
            "endpoint_tick_alignment": "outer plot and colorbar labels pulled inward",
            "y_label_alignment": "vertical center of the MSS axes",
            "pdf_sha256": _sha256(figure_path),
        },
        "source_sha256": {
            "scripts/render_loss_landscapes.py": _sha256(Path(__file__).resolve()),
            "src/config.py": _sha256(ROOT / "src" / "config.py"),
            "src/exciter.py": _sha256(ROOT / "src" / "exciter.py"),
            "src/losses.py": _sha256(ROOT / "src" / "losses.py"),
            "src/synth.py": _sha256(ROOT / "src" / "synth.py"),
            "src/waveguide.py": _sha256(ROOT / "src" / "waveguide.py"),
        },
    }
    provenance_path = output_stem.with_suffix(".provenance.json")
    _atomic_json(provenance_path, payload)
    return {
        "figure": str(figure_path),
        "provenance": str(provenance_path),
        "diagnostics": diagnostics,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract-qualified")
    extract.add_argument(
        "--source-pdf", type=Path, default=DEFAULT_OUTPUT_STEM.with_suffix(".pdf")
    )
    extract.add_argument("--output", type=Path, default=DEFAULT_QUALIFIED_CACHE)
    comparison = commands.add_parser("compute-comparison")
    comparison.add_argument("--device", default="cuda")
    comparison.add_argument("--gradient-chunk-size", type=int, default=16)
    comparison.add_argument("--output", type=Path, default=DEFAULT_COMPARISON_CACHE)
    build = commands.add_parser("build")
    build.add_argument("--qualified-cache", type=Path, default=DEFAULT_QUALIFIED_CACHE)
    build.add_argument("--comparison-cache", type=Path, default=DEFAULT_COMPARISON_CACHE)
    build.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "extract-qualified":
        report = extract_qualified_cache(args.source_pdf.resolve(), args.output.resolve())
    elif args.command == "compute-comparison":
        report = compute_comparison_cache(
            args.device, args.gradient_chunk_size, args.output.resolve()
        )
    else:
        report = build_figure(
            args.qualified_cache.resolve(),
            args.comparison_cache.resolve(),
            args.output_stem,
        )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
