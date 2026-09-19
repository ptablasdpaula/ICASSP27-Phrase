"""Eight-loss renderer-sensitivity data for the supplementary website.

Each diagnostic cell matches the target and candidate renderer, then evaluates
all eight paper losses on the registered single-event lattice.  Recursive
rendering is deliberately CUDA-only; the browser consumes precomputed JSON.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import Tensor

from .config import ExciterConfig, WaveguideConfig
from .losses import LOSS_LABELS, LOSS_NAMES, build_loss
from .runtime import configure_reproducibility, require_accelerated_backend
from .synth import PhraseSynth

GRID_POINTS: Final = 25
TARGET_INDEX: Final = GRID_POINTS // 2
TARGET_F0_HZ: Final = 160.0
TARGET_ONSET_SECONDS: Final = 1.0
DISPLAY_INDICES: Final = tuple(range(2, 23, 2))
ONSET_METHODS: Final = ("naive", "lagrange", "fourier", "thiran")
PROPAGATION_METHODS: Final = ("linear", "lagrange", "fourier", "thiran")
ONSET_LABELS: Final = {
    "naive": "Analytic",
    "lagrange": "Lagrange-5",
    "fourier": "Fourier",
    "thiran": "Thiran-1",
}
PROPAGATION_LABELS: Final = {
    "linear": "Linear",
    "lagrange": "Lagrange-5/1",
    "fourier": "Fourier",
    "thiran": "Thiran-3/1",
}
CELL_SCHEMA: Final = "renderer-screen-eight-loss-cell-v1"
LOSS_DATA_SCHEMA: Final = "renderer-screen-loss-data-v1"
MANIFEST_SCHEMA: Final = "renderer-screen-eight-loss-manifest-v1"


@dataclass(frozen=True)
class RendererCell:
    """One matched onset-placement and propagation combination."""

    index: int
    onset: str
    propagation: str

    @property
    def key(self) -> str:
        return f"{self.onset}__{self.propagation}"

    @property
    def exciter(self) -> ExciterConfig:
        return ExciterConfig(method=self.onset)

    @property
    def waveguide(self) -> WaveguideConfig:
        return WaveguideConfig(
            interpolation=self.propagation,
            realization="frequency" if self.propagation == "fourier" else "df2",
            state_policy="hard_reset",
        )


def renderer_cells() -> tuple[RendererCell, ...]:
    """Return the website's row-major 4-by-4 renderer matrix."""
    return tuple(
        RendererCell(index, onset, propagation)
        for index, (onset, propagation) in enumerate(
            (onset, propagation) for onset in ONSET_METHODS for propagation in PROPAGATION_METHODS
        )
    )


def renderer_cell(index: int) -> RendererCell:
    cells = renderer_cells()
    if index < 0 or index >= len(cells):
        raise ValueError("renderer cell index must lie in [0,15]")
    return cells[index]


def coordinate_axes() -> tuple[np.ndarray, np.ndarray]:
    """Return the registered normalized pitch and onset axes."""
    pitch = np.linspace(0.0, 1.0, GRID_POINTS, dtype=np.float64)
    onset = np.linspace(0.0, 1.0, GRID_POINTS, dtype=np.float64)
    pitch[TARGET_INDEX] = 0.5
    onset[TARGET_INDEX] = 0.5
    return pitch, onset


def surface_coordinates() -> np.ndarray:
    """Return all 625 pitch/onset coordinates in pitch-major order."""
    pitch, onset = coordinate_axes()
    pitch_mesh, onset_mesh = np.meshgrid(pitch, onset, indexing="ij")
    result = np.stack((pitch_mesh.reshape(-1), onset_mesh.reshape(-1)), axis=-1)
    if result.shape != (GRID_POINTS**2, 2):
        raise RuntimeError("registered surface grid is malformed")
    return result


def displayed_coordinates() -> np.ndarray:
    """Return the same 120 non-target coordinates displayed in Figure 1."""
    pitch, onset = coordinate_axes()
    result = np.asarray(
        [
            (pitch[row], onset[column])
            for row in DISPLAY_INDICES
            for column in DISPLAY_INDICES
            if (row, column) != (TARGET_INDEX, TARGET_INDEX)
        ],
        dtype=np.float64,
    )
    if result.shape != (120, 2):
        raise RuntimeError("direction sublattice must contain 120 points")
    return result


def physical_controls(coordinates: Tensor) -> tuple[Tensor, Tensor]:
    """Map normalized pitch/onset coordinates to registered controls."""
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape [batch,2]")
    f0_hz = 80.0 * torch.pow(4.0, coordinates[:, 0:1])
    onset_seconds = 0.2 + 1.6 * coordinates[:, 1:2]
    return f0_hz, onset_seconds


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha256(payload: dict[str, Any]) -> str:
    value = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any], *, sign: bool = False) -> None:
    value = dict(payload)
    if sign:
        value["payload_sha256"] = _payload_sha256(payload)
    serialized = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _target_audio(synth: PhraseSynth, device: torch.device) -> Tensor:
    coordinate = torch.tensor([[0.5, 0.5]], dtype=torch.float64, device=device)
    with torch.no_grad():
        f0_hz, onset_seconds = physical_controls(coordinate)
        target = synth(f0_hz, onset_seconds)[0].detach()
    if target.shape != (32_000,) or not bool(torch.isfinite(target).all()):
        raise FloatingPointError("renderer produced a malformed target")
    return target


def _surface_pass(
    synth: PhraseSynth,
    metrics: list[Any],
    *,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    coordinates = surface_coordinates()
    values: list[list[np.ndarray]] = [[] for _ in LOSS_NAMES]
    with torch.no_grad():
        for begin in range(0, coordinates.shape[0], chunk_size):
            batch = torch.tensor(
                coordinates[begin : begin + chunk_size],
                dtype=torch.float64,
                device=device,
            )
            f0_hz, onset_seconds = physical_controls(batch)
            audio = synth(f0_hz, onset_seconds)
            if not bool(torch.isfinite(audio).all()):
                raise FloatingPointError("renderer produced non-finite surface audio")
            for loss_index, metric in enumerate(metrics):
                distance = metric.distances(audio)
                if not bool(torch.isfinite(distance).all()):
                    raise FloatingPointError(f"{LOSS_NAMES[loss_index]} surface is non-finite")
                values[loss_index].append(distance.cpu().numpy())
    result = np.stack([np.concatenate(rows).reshape(GRID_POINTS, GRID_POINTS) for rows in values])
    if result.shape != (len(LOSS_NAMES), GRID_POINTS, GRID_POINTS):
        raise RuntimeError("surface pass produced an unexpected shape")
    return result


def _direction_pass(
    synth: PhraseSynth,
    metrics: list[Any],
    *,
    device: torch.device,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    positions = displayed_coordinates()
    chunks: list[list[np.ndarray]] = [[] for _ in LOSS_NAMES]
    for begin in range(0, positions.shape[0], chunk_size):
        coordinates = torch.tensor(
            positions[begin : begin + chunk_size],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        f0_hz, onset_seconds = physical_controls(coordinates)
        audio = synth(f0_hz, onset_seconds)
        if not bool(torch.isfinite(audio.detach()).all()):
            raise FloatingPointError("renderer produced non-finite direction audio")
        for loss_index, metric in enumerate(metrics):
            distance = metric.distances(audio)
            if not bool(torch.isfinite(distance.detach()).all()):
                raise FloatingPointError(f"{LOSS_NAMES[loss_index]} direction loss is non-finite")
            (gradient,) = torch.autograd.grad(
                distance.sum(),
                coordinates,
                retain_graph=loss_index < len(metrics) - 1,
            )
            if not bool(torch.isfinite(gradient.detach()).all()):
                raise FloatingPointError(f"{LOSS_NAMES[loss_index]} direction is non-finite")
            chunks[loss_index].append(-gradient.detach().cpu().numpy())

    raw = np.stack([np.concatenate(rows) for rows in chunks])
    norms = np.linalg.norm(raw, axis=-1, keepdims=True)
    unit = np.divide(raw, norms, out=np.zeros_like(raw), where=norms > 0.0)
    target_displacement = 0.5 - positions
    alignment = np.sum(unit * target_displacement[None], axis=-1)
    counts = np.count_nonzero(alignment > 0.0, axis=1).astype(np.int64)
    if unit.shape != (len(LOSS_NAMES), 120, 2) or counts.shape != (len(LOSS_NAMES),):
        raise RuntimeError("direction pass produced an unexpected shape")
    return unit, counts


def compute_cell(
    index: int,
    output_root: Path,
    *,
    device_name: str = "cuda",
    surface_chunk_size: int = 16,
    gradient_chunk_size: int = 4,
) -> dict[str, Any]:
    """Compute all eight losses for one matched renderer cell on CUDA."""
    if surface_chunk_size < 1 or gradient_chunk_size < 1:
        raise ValueError("chunk sizes must be positive")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError("renderer-screen waveform evaluation is CUDA-only")
    configure_reproducibility()
    require_accelerated_backend(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()

    cell = renderer_cell(index)
    synth = PhraseSynth(
        exciter_config=cell.exciter,
        waveguide_config=cell.waveguide,
    ).to(device)
    target = _target_audio(synth, device)
    metrics = [build_loss(name, target) for name in LOSS_NAMES]
    losses = _surface_pass(synth, metrics, device=device, chunk_size=surface_chunk_size)
    directions, counts = _direction_pass(
        synth, metrics, device=device, chunk_size=gradient_chunk_size
    )
    torch.cuda.synchronize(device)

    low = losses.min(axis=(1, 2))
    high = losses.max(axis=(1, 2))
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)) or np.any(high <= low):
        raise FloatingPointError("one or more loss surfaces has no finite dynamic range")
    normalized = (losses - low[:, None, None]) / (high - low)[:, None, None]

    output_root = output_root.resolve()
    npz_path = output_root / "cells" / f"{cell.key}.npz"
    _atomic_npz(
        npz_path,
        schema=np.asarray(CELL_SCHEMA),
        loss_names=np.asarray(LOSS_NAMES),
        loss_labels=np.asarray(LOSS_LABELS),
        normalized_loss=normalized,
        raw_loss=losses,
        unit_descent=directions,
        target_directed_counts=counts,
        displayed_coordinates=displayed_coordinates(),
    )
    root = Path(__file__).resolve().parents[1]
    payload = {
        "schema": CELL_SCHEMA,
        "source_commit": _git_commit(root),
        "cell": {
            "index": cell.index,
            "key": cell.key,
            "onset": cell.onset,
            "onset_label": ONSET_LABELS[cell.onset],
            "propagation": cell.propagation,
            "propagation_label": PROPAGATION_LABELS[cell.propagation],
        },
        "renderer": {
            "exciter": asdict(cell.exciter),
            "waveguide": asdict(cell.waveguide),
            "target_candidate_matching": "same renderer within cell",
        },
        "losses": [
            {
                "id": name,
                "label": label,
                "target_directed_count": int(count),
                "target_directed_percentage": round(100.0 * int(count) / 120.0, 1),
                "raw_minimum": float(minimum),
                "raw_maximum": float(maximum),
            }
            for name, label, count, minimum, maximum in zip(
                LOSS_NAMES, LOSS_LABELS, counts, low, high, strict=True
            )
        ],
        "protocol": protocol_payload(),
        "runtime": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "surface_chunk_size": surface_chunk_size,
            "gradient_chunk_size": gradient_chunk_size,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
            "seconds": time.monotonic() - started,
        },
        "npz": {"path": str(npz_path), "sha256": _sha256(npz_path)},
    }
    metadata_path = npz_path.with_suffix(".json")
    _atomic_json(metadata_path, payload, sign=True)
    return {**payload, "metadata_path": str(metadata_path)}


def protocol_payload() -> dict[str, Any]:
    """Return the shared, JSON-safe registered diagnostic description."""
    return {
        "target": {"f0_hz": TARGET_F0_HZ, "onset_seconds": TARGET_ONSET_SECONDS},
        "surface_grid": {
            "shape": [GRID_POINTS, GRID_POINTS],
            "f0_hz": [80.0, 320.0],
            "onset_seconds": [0.2, 1.8],
            "pitch_coordinate": "log2(f0/80)/2",
            "onset_coordinate": "(onset-0.2)/1.6",
            "normalization": "independent min-max per loss and renderer cell",
        },
        "directions": {
            "display_indices": list(DISPLAY_INDICES),
            "displayed_non_target_gradients": 120,
            "criterion": "dot(-unit_gradient, target_minus_candidate) > 0",
            "coordinates": "normalized log-pitch and linear onset",
        },
    }


def _load_cell(
    input_root: Path, cell: RendererCell
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    npz_path = input_root / "cells" / f"{cell.key}.npz"
    metadata_path = npz_path.with_suffix(".json")
    if not npz_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"renderer-screen cell {cell.key} is incomplete")
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata.get("schema") != CELL_SCHEMA
        or metadata.get("cell", {}).get("index") != cell.index
        or metadata.get("cell", {}).get("key") != cell.key
        or metadata.get("npz", {}).get("sha256") != _sha256(npz_path)
    ):
        raise ValueError(f"renderer-screen cell {cell.key} failed identity validation")
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if (
        str(arrays.get("schema", "")) != CELL_SCHEMA
        or tuple(arrays.get("loss_names", ())) != LOSS_NAMES
        or arrays["normalized_loss"].shape != (len(LOSS_NAMES), 25, 25)
        or arrays["unit_descent"].shape != (len(LOSS_NAMES), 120, 2)
        or arrays["target_directed_counts"].shape != (len(LOSS_NAMES),)
    ):
        raise ValueError(f"renderer-screen cell {cell.key} has malformed arrays")
    return metadata, arrays


def _rounded(value: np.ndarray, decimals: int = 7) -> list[Any]:
    return np.round(value.astype(np.float64), decimals=decimals).tolist()


def assemble_web_data(input_root: Path, output_root: Path) -> dict[str, Any]:
    """Assemble compact, lazy-loadable browser data from all sixteen cells."""
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    loaded = [(cell, *_load_cell(input_root, cell)) for cell in renderer_cells()]
    commits = {metadata["source_commit"] for _, metadata, _ in loaded}
    if len(commits) != 1:
        raise ValueError("renderer-screen cells came from different source commits")
    source_commit = commits.pop()
    positions = displayed_coordinates()
    files: dict[str, dict[str, Any]] = {}

    for loss_index, (loss_name, loss_label) in enumerate(zip(LOSS_NAMES, LOSS_LABELS, strict=True)):
        cells = []
        for cell, metadata, arrays in loaded:
            count = int(arrays["target_directed_counts"][loss_index])
            loss_metadata = metadata["losses"][loss_index]
            directions = arrays["unit_descent"][loss_index]
            arrows = np.concatenate((positions, directions), axis=1)
            cells.append(
                {
                    "index": cell.index,
                    "key": cell.key,
                    "onset": cell.onset,
                    "onset_label": ONSET_LABELS[cell.onset],
                    "propagation": cell.propagation,
                    "propagation_label": PROPAGATION_LABELS[cell.propagation],
                    "target_directed_count": count,
                    "target_directed_percentage": round(100.0 * count / 120.0, 1),
                    "raw_loss_range": [
                        loss_metadata["raw_minimum"],
                        loss_metadata["raw_maximum"],
                    ],
                    "normalized_loss": _rounded(arrays["normalized_loss"][loss_index].reshape(-1)),
                    "arrows": _rounded(arrows),
                }
            )
        payload = {
            "schema": LOSS_DATA_SCHEMA,
            "source_commit": source_commit,
            "loss": {"id": loss_name, "label": loss_label, "index": loss_index},
            "protocol": protocol_payload(),
            "cells": cells,
        }
        path = output_root / f"{loss_name}.json"
        _atomic_json(path, payload, sign=True)
        files[loss_name] = {
            "path": path.name,
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "source_commit": source_commit,
        "losses": [
            {"id": name, "label": label, **files[name]}
            for name, label in zip(LOSS_NAMES, LOSS_LABELS, strict=True)
        ],
        "rows": [{"id": name, "label": ONSET_LABELS[name]} for name in ONSET_METHODS],
        "columns": [
            {"id": name, "label": PROPAGATION_LABELS[name]} for name in PROPAGATION_METHODS
        ],
        "protocol": protocol_payload(),
        "cell_artifacts": {
            cell.key: {
                "npz_sha256": metadata["npz"]["sha256"],
                "payload_sha256": metadata["payload_sha256"],
            }
            for cell, metadata, _ in loaded
        },
    }
    manifest_path = output_root / "manifest.json"
    _atomic_json(manifest_path, manifest, sign=True)
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "source_commit": source_commit,
        "files": files,
    }


__all__ = [
    "CELL_SCHEMA",
    "DISPLAY_INDICES",
    "GRID_POINTS",
    "LOSS_DATA_SCHEMA",
    "MANIFEST_SCHEMA",
    "ONSET_LABELS",
    "ONSET_METHODS",
    "PROPAGATION_LABELS",
    "PROPAGATION_METHODS",
    "RendererCell",
    "assemble_web_data",
    "compute_cell",
    "coordinate_axes",
    "displayed_coordinates",
    "physical_controls",
    "protocol_payload",
    "renderer_cell",
    "renderer_cells",
    "surface_coordinates",
]
