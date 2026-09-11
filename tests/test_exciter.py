from __future__ import annotations

import pytest
import torch

from icassp27_phrase import Exciter, ExciterConfig


@pytest.mark.parametrize("method", ("naive", "lagrange", "fourier", "thiran"))
def test_integer_onset_methods_equal_the_sampled_hrc(method: str) -> None:
    config = ExciterConfig(method=method, sample_count=512, fourier_fft_length=2048)
    onset = torch.tensor([[0.05]], dtype=torch.float64)
    actual = Exciter(config)(onset)
    expected = Exciter(ExciterConfig(method="naive", sample_count=512))(onset)
    tolerance = 2e-12 if method != "fourier" else 3e-12
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=tolerance)


@pytest.mark.parametrize("method", ("naive", "lagrange", "fourier", "thiran"))
def test_onset_gradient_is_finite_and_nonzero(method: str) -> None:
    config = ExciterConfig(method=method, sample_count=512, fourier_fft_length=2048)
    onset = torch.tensor([[0.05013]], dtype=torch.float64, requires_grad=True)
    audio = Exciter(config)(onset)
    probe = torch.linspace(-1.0, 1.0, audio.shape[-1], dtype=torch.float64)
    gradient, = torch.autograd.grad((audio * probe).sum(), onset)
    assert bool(torch.isfinite(gradient).all())
    assert float(gradient.abs().sum()) > 0.0


def test_amplitude_and_duration_are_construction_controls() -> None:
    onset = torch.zeros((1, 1), dtype=torch.float64)
    short = Exciter(
        ExciterConfig(method="naive", sample_count=128, amplitude=0.4, duration_seconds=0.005)
    )(onset)
    short_full_amplitude = Exciter(
        ExciterConfig(method="naive", sample_count=128, amplitude=0.8, duration_seconds=0.005)
    )(onset)
    paper = Exciter(ExciterConfig(method="naive", sample_count=128))(onset)
    torch.testing.assert_close(short_full_amplitude, 2.0 * short, rtol=0.0, atol=0.0)
    assert float(short.max()) < 0.4
    assert float(paper.max()) < 0.8
    assert int(torch.count_nonzero(short)) < int(torch.count_nonzero(paper))
