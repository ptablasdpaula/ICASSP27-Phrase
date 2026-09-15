from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from icassp27_phrase.losses import LOSS_NAMES
from icassp27_phrase.renderer_screen import (
    CELL_SCHEMA,
    LOSS_DATA_SCHEMA,
    MANIFEST_SCHEMA,
    assemble_web_data,
    compute_cell,
    displayed_coordinates,
    renderer_cell,
    renderer_cells,
    surface_coordinates,
)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_registered_renderer_matrix_and_grids() -> None:
    cells = renderer_cells()
    assert len(cells) == 16
    assert len({cell.key for cell in cells}) == 16
    assert cells[0].key == "naive__linear"
    assert cells[-1].key == "thiran__thiran"
    assert renderer_cell(6).key == "lagrange__fourier"
    assert renderer_cell(2).waveguide.realization == "frequency"
    assert renderer_cell(3).waveguide.realization == "df2"
    assert surface_coordinates().shape == (625, 2)
    assert displayed_coordinates().shape == (120, 2)
    assert not np.any(np.all(displayed_coordinates() == 0.5, axis=1))


def test_compute_cell_refuses_cpu(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="CUDA-only"):
        compute_cell(0, tmp_path, device_name="cpu")


def test_assemble_web_data_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    positions = displayed_coordinates()
    monkeypatch.setattr("icassp27_phrase.renderer_screen._git_commit", lambda _root: "a" * 40)
    for cell in renderer_cells():
        path = input_root / "cells" / f"{cell.key}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        surface = np.linspace(0.0, 1.0, 625).reshape(25, 25)
        normalized = np.stack([surface for _ in LOSS_NAMES])
        directions = np.zeros((len(LOSS_NAMES), 120, 2), dtype=np.float64)
        directions[..., 0] = np.sign(0.5 - positions[:, 0])
        counts = np.arange(8, dtype=np.int64) + 80
        np.savez_compressed(
            path,
            schema=np.asarray(CELL_SCHEMA),
            loss_names=np.asarray(LOSS_NAMES),
            loss_labels=np.asarray(tuple(f"loss-{index}" for index in range(8))),
            normalized_loss=normalized,
            raw_loss=normalized,
            unit_descent=directions,
            target_directed_counts=counts,
            displayed_coordinates=positions,
        )
        losses = [
            {
                "id": name,
                "label": f"loss-{index}",
                "target_directed_count": int(counts[index]),
                "target_directed_percentage": round(100 * int(counts[index]) / 120, 1),
                "raw_minimum": 0.0,
                "raw_maximum": 1.0,
            }
            for index, name in enumerate(LOSS_NAMES)
        ]
        metadata = {
            "schema": CELL_SCHEMA,
            "source_commit": "a" * 40,
            "cell": {"index": cell.index, "key": cell.key},
            "losses": losses,
            "npz": {"sha256": _sha256(path)},
            "payload_sha256": f"cell-{cell.index}",
        }
        path.with_suffix(".json").write_text(json.dumps(metadata))

    report = assemble_web_data(input_root, output_root)
    manifest = json.loads(Path(report["manifest"]).read_text())
    assert manifest["schema"] == MANIFEST_SCHEMA
    assert len(manifest["losses"]) == 8
    assert len(manifest["cell_artifacts"]) == 16
    first = json.loads((output_root / f"{LOSS_NAMES[0]}.json").read_text())
    assert first["schema"] == LOSS_DATA_SCHEMA
    assert len(first["cells"]) == 16
    assert len(first["cells"][0]["normalized_loss"]) == 625
    assert len(first["cells"][0]["arrows"]) == 120
