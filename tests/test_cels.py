from __future__ import annotations

import itertools

import pytest
import torch

from cels import (
    ALL_DIRECTIONS,
    CumulativeEnergyLoss,
    Direction,
    STFTCumulativeEnergyLoss,
    paper_loss,
)


def explicit_surface(x: torch.Tensor, direction: Direction) -> torch.Tensor:
    f, t = x.shape
    result = torch.zeros_like(x)
    for k, n in itertools.product(range(f), range(t)):
        if direction == Direction.RIGHT_UP:
            result[k, n] = x[: k + 1, : n + 1].sum()
        elif direction == Direction.RIGHT_DOWN:
            result[k, n] = x[k:, : n + 1].sum()
        elif direction == Direction.LEFT_UP:
            result[k, n] = x[: k + 1, n:].sum()
        elif direction == Direction.LEFT_DOWN:
            result[k, n] = x[k:, n:].sum()
        elif direction == Direction.UP:
            result[k, n] = x[: k + 1, n].sum()
        elif direction == Direction.DOWN:
            result[k, n] = x[k:, n].sum()
        elif direction == Direction.RIGHT:
            result[k, n] = x[k, : n + 1].sum()
        else:
            result[k, n] = x[k, n:].sum()
    return result


@pytest.mark.parametrize("direction", ALL_DIRECTIONS)
def test_all_directions_match_explicit_sums(direction: Direction) -> None:
    target = torch.arange(1, 13, dtype=torch.float64).reshape(3, 4)
    candidate = target.flip((0, 1))
    criterion = CumulativeEnergyLoss(directions=(direction,), reduction="none")
    actual = criterion(candidate, target)
    mass = target.sum()
    expected_error = (
        (explicit_surface(candidate, direction) / mass).clamp_min(1e-12).sqrt()
        - (explicit_surface(target, direction) / mass).clamp_min(1e-12).sqrt()
    )
    torch.testing.assert_close(actual, expected_error.square().mean().sqrt())


def test_subset_average_broadcast_and_bound_equivalence() -> None:
    target = torch.arange(1, 13, dtype=torch.float64).reshape(3, 4)
    candidates = torch.stack((target, target.flip(0), target.flip(1))).requires_grad_()
    directions = (Direction.RIGHT_UP, Direction.DOWN, Direction.LEFT)
    criterion = CumulativeEnergyLoss(directions=directions, reduction="none")
    terms = criterion.directional_distances(candidates, target)
    torch.testing.assert_close(criterion(candidates, target), terms.mean(-1))
    torch.testing.assert_close(criterion.bind(target)(candidates), terms.mean(-1))
    assert criterion(candidates, target)[0] == 0
    criterion(candidates, target).sum().backward()
    assert torch.isfinite(candidates.grad).all()


def test_log_weighting_and_decay_combine() -> None:
    target = torch.arange(1, 13, dtype=torch.float64).reshape(3, 4)
    candidate = target.flip(0).requires_grad_()
    frequency = torch.tensor([0.0, 100.0, 400.0], dtype=torch.float64)
    time = torch.tensor([0.0, 0.1, 0.4, 1.0], dtype=torch.float64)
    criterion = CumulativeEnergyLoss(
        directions=ALL_DIRECTIONS,
        log_weighting=True,
        time_decay_seconds=0.5,
        frequency_decay_hz=250.0,
    )
    value = criterion(candidate, target, frequency=frequency, time=time)
    assert torch.isfinite(value) and value > 0
    value.backward()
    assert torch.isfinite(candidate.grad).all()
    torch.testing.assert_close(
        value,
        criterion.bind(target, frequency=frequency, time=time)(candidate.detach()),
    )


def test_decay_has_zero_weight_at_and_beyond_horizon() -> None:
    target = torch.ones((1, 3), dtype=torch.float64)
    candidate = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float64)
    time = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    loss = CumulativeEnergyLoss(
        directions=(Direction.RIGHT,), time_decay_seconds=1.0, reduction="none"
    )
    # At the last output bin, the first input is beyond the horizon and the
    # second is exactly at it; both have zero contribution.
    geometry = loss._geometry(target, None, time)
    surface = torch.einsum("...fi,oi->...fo", candidate, geometry.time_forward)
    torch.testing.assert_close(surface[0, -1], candidate[0, -1])


def test_stft_wrapper_and_paper_presets() -> None:
    torch.manual_seed(1)
    target = torch.randn(4096, dtype=torch.float64)
    candidate = (target + 0.01 * torch.randn_like(target)).requires_grad_()
    loss = STFTCumulativeEnergyLoss(sample_rate=16_000, n_fft=256, hop_length=64)
    direct = loss(candidate, target)
    bound = loss.bind(target)(candidate)
    torch.testing.assert_close(direct, bound)
    direct.backward()
    assert torch.isfinite(candidate.grad).all()
    for variant in ("cel", "log_cel", "dec_cel", "tlog_cel"):
        assert isinstance(paper_loss(variant), STFTCumulativeEnergyLoss)


def test_invalid_inputs_are_rejected() -> None:
    target = torch.ones((3, 4), dtype=torch.float64)
    with pytest.raises(ValueError):
        CumulativeEnergyLoss(directions=())
    with pytest.raises(ValueError):
        CumulativeEnergyLoss(directions=(Direction.UP, Direction.UP))
    with pytest.raises(ValueError):
        CumulativeEnergyLoss(log_weighting=True)(target, target)
    with pytest.raises(ValueError):
        CumulativeEnergyLoss()(target, torch.zeros_like(target))
    with pytest.raises(ValueError):
        CumulativeEnergyLoss()(target.neg(), target)
