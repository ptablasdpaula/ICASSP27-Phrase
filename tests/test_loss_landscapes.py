from __future__ import annotations

import numpy as np
import pytest

from scripts.render_loss_landscapes import (
    LANDSCAPE_FIGURE_STYLE,
    TARGET_INDEX,
    compute_comparison_cache,
    coordinate_grids,
    displayed_coordinates,
    renderer_specs,
    target_direction_summary,
)


def test_landscape_grid_has_one_exact_central_target() -> None:
    pitch, onset, coordinates = coordinate_grids()
    assert coordinates.shape == (625, 2)
    assert pitch[TARGET_INDEX] == 0.5
    assert onset[TARGET_INDEX] == 0.5
    assert np.count_nonzero(pitch == 0.5) == 1
    assert np.count_nonzero(onset == 0.5) == 1


def test_display_sublattice_excludes_target_and_has_120_points() -> None:
    coordinates = displayed_coordinates()
    assert coordinates.shape == (120, 2)
    assert not np.any(np.all(coordinates == np.asarray([0.5, 0.5]), axis=1))


def test_target_direction_summary_distinguishes_toward_and_away() -> None:
    coordinates = displayed_coordinates()
    toward = 0.5 - coordinates
    toward /= np.linalg.norm(toward, axis=1, keepdims=True)
    assert target_direction_summary(coordinates, toward) == {
        "target_directed_count": 120,
        "displayed_non_target_gradients": 120,
        "target_directed_percentage": 100.0,
    }
    assert target_direction_summary(coordinates, -toward) == {
        "target_directed_count": 0,
        "displayed_non_target_gradients": 120,
        "target_directed_percentage": 0.0,
    }


def test_renderer_comparison_is_fourier_thiran_and_naive_linear() -> None:
    campaign, comparison = renderer_specs()
    assert (campaign.exciter.method, campaign.waveguide.interpolation) == (
        "fourier",
        "thiran",
    )
    assert (comparison.exciter.method, comparison.waveguide.interpolation) == (
        "naive",
        "linear",
    )
    assert campaign.waveguide.state_policy == comparison.waveguide.state_policy == "hard_reset"


def test_landscape_figure_restores_registered_white_compact_style() -> None:
    style = LANDSCAPE_FIGURE_STYLE
    assert style.arrow_color == style.annotation_color == "white"
    assert style.arrow_half_length == 0.026
    assert style.arrow_mutation_scale == 4.2
    assert style.arrow_linewidth == 0.48
    assert style.annotation_y == 0.985
    assert style.horizontal_space == 0.065
    assert style.horizontal_center == pytest.approx(
        0.5 * (style.plot_left + style.plot_right)
    )


def test_recursive_comparison_rejects_cpu(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="must run on CUDA"):
        compute_comparison_cache("cpu", 1, tmp_path / "comparison.npz")
