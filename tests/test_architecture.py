from __future__ import annotations

from pathlib import Path

import pytest
import torch

from icassp27_phrase import (
    Exciter,
    ExciterConfig,
    PhraseSynth,
    Waveguide,
    WaveguideConfig,
    initial_candidate,
    load_target,
)


def test_public_synth_is_exactly_exciter_then_waveguide() -> None:
    synth = PhraseSynth()
    assert isinstance(synth.exciter, Exciter)
    assert isinstance(synth.waveguide, Waveguide)
    assert synth.exciter.config.method == "fourier"
    assert synth.waveguide.config == WaveguideConfig(
        interpolation="thiran", realization="df2", state_policy="hard_reset"
    )
    assert synth.provenance()["architecture"] == ["Exciter", "Waveguide", "PhraseSynth"]


@pytest.mark.parametrize("cardinality", (1, 2, 4, 6, 8))
def test_registered_initial_candidate(cardinality: int) -> None:
    phrase = initial_candidate(cardinality, device="cpu")
    torch.testing.assert_close(
        phrase.f0_hz, torch.full((cardinality,), 160.0, dtype=torch.float64)
    )
    index = torch.arange(cardinality, dtype=torch.float64)
    expected = 0.2 + (index + 0.5) * (1.6 / cardinality)
    torch.testing.assert_close(phrase.onset_seconds, expected, rtol=0.0, atol=0.0)


def test_all_150_targets_load_for_every_cardinality() -> None:
    for cardinality in (1, 2, 4, 6, 8):
        for index in range(1, 151):
            metadata, phrase = load_target(cardinality, index)
            assert metadata.index == index
            assert phrase.cardinality == cardinality
    first, _ = load_target(1, 1)
    assert first.f0_hz[0] == pytest.approx(319.6450634016123, rel=0.0, abs=0.0)
    assert first.onset_seconds[0] == pytest.approx(0.6182587918583865, rel=0.0, abs=0.0)


def test_repository_has_no_legacy_renderer_name() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in [*root.joinpath("src").rglob("*.py"), *root.joinpath("paper").rglob("*.tex")]:
        assert "simple" + "dwg" not in path.read_text(encoding="utf-8").lower()


def test_invalid_combinations_fail_at_construction() -> None:
    with pytest.raises(ValueError, match="persistent state"):
        WaveguideConfig(
            interpolation="fourier", realization="frequency", state_policy="persistent"
        )
    with pytest.raises(ValueError, match="frequency realization"):
        WaveguideConfig(interpolation="fourier", realization="literal")
    with pytest.raises(ValueError, match="orders"):
        ExciterConfig(method="lagrange", lagrange_order=3)
