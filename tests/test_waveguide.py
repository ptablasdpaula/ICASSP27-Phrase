from __future__ import annotations

import pytest
import torch
from icassp27_phrase import ExciterConfig, PhraseSynth, Waveguide, WaveguideConfig
from icassp27_phrase.waveguide import (
    _literal_two_rail_scan,
    _pad_event_coefficients,
    _segment_rows,
)


@pytest.mark.parametrize("method", ("linear", "lagrange", "thiran"))
def test_literal_and_collapsed_static_realizations_are_equivalent(method: str) -> None:
    config = WaveguideConfig(
        interpolation=method, realization="literal", sample_count=512
    )
    frequency = torch.tensor([173.2], dtype=torch.float64)
    source = torch.zeros((1, config.sample_count), dtype=torch.float64)
    source[:, 0] = 1.0
    collapsed = Waveguide(
        WaveguideConfig(interpolation=method, realization="df2", sample_count=512)
    )(
        source,
        frequency[None],
        torch.tensor([[0.2]], dtype=torch.float64),
    )
    long, short, _ = _segment_rows(frequency, config)
    long_width = max(long.numerator.shape[-1], long.denominator.shape[-1])
    short_width = max(short.numerator.shape[-1], short.denominator.shape[-1])
    literal = _literal_two_rail_scan(
        source,
        _pad_event_coefficients(long.numerator[:, None], long_width),
        _pad_event_coefficients(long.denominator[:, None], long_width),
        _pad_event_coefficients(short.numerator[:, None], short_width),
        _pad_event_coefficients(short.denominator[:, None], short_width),
        torch.zeros((1, 1), dtype=torch.long),
        config.loop_gain,
        config.loop_pole,
        False,
    )
    torch.testing.assert_close(literal, collapsed, rtol=4e-10, atol=3e-11)


@pytest.mark.parametrize(
    ("method", "orders"),
    (("linear", (1, 1)), ("lagrange", (5, 1)), ("thiran", (3, 3))),
)
def test_registered_segment_orders_and_gradients(method: str, orders: tuple[int, int]) -> None:
    config = WaveguideConfig(interpolation=method)
    assert config.segment_orders == orders
    frequency = torch.tensor([173.2], dtype=torch.float64, requires_grad=True)
    long, short, rail = _segment_rows(frequency, config)
    scalar = long.numerator.square().sum() + short.numerator.square().sum() + rail.sum()
    gradient, = torch.autograd.grad(scalar, frequency)
    assert bool(torch.isfinite(gradient).all())
    assert float(gradient.abs().sum()) > 0.0


def test_paper_thiran_transfer_has_expected_physical_lengths() -> None:
    transfer = Waveguide().transfer(torch.tensor(160.0, dtype=torch.float64))
    assert float(transfer.long_delay_samples) == pytest.approx(38.404, abs=0.01)
    assert float(transfer.short_delay_samples) == pytest.approx(11.471, abs=0.01)


def test_hard_reset_paper_path_has_cpu_autograd() -> None:
    sample_count = 4_096
    synth = PhraseSynth(
        exciter_config=ExciterConfig(
            method="fourier",
            sample_count=sample_count,
            fourier_fft_length=8_192,
        ),
        waveguide_config=WaveguideConfig(sample_count=sample_count),
    )
    frequency = torch.tensor([[173.2]], dtype=torch.float64, requires_grad=True)
    onset = torch.tensor([[0.20013]], dtype=torch.float64, requires_grad=True)
    audio = synth.render_batch(frequency, onset)
    probe = torch.linspace(-1.0, 1.0, sample_count, dtype=torch.float64)
    gradients = torch.autograd.grad((audio * probe).sum(), (frequency, onset))
    assert audio.shape == (1, sample_count)
    assert bool(torch.isfinite(audio).all())
    for gradient in gradients:
        assert bool(torch.isfinite(gradient).all())
        assert float(gradient.abs().sum()) > 0.0
