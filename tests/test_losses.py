from __future__ import annotations

import torch
from icassp27_phrase.losses import (
    CEL_HOP,
    CEL_N_FFT,
    DDSP_FFT_SIZES,
    NAMES,
    SMOOTH_HOPS,
    SMOOTH_WINDOWS,
    SOT_HOP,
    SOT_N_FFT,
    TFW_HOP,
    TFW_N_FFT,
    PaperObjectives,
)


def test_frozen_16khz_loss_configurations() -> None:
    assert len(NAMES) == len(set(NAMES)) == 13
    assert (CEL_N_FFT, CEL_HOP) == (1024, 256)
    assert DDSP_FFT_SIZES == (2048, 1024, 512, 256, 128, 64)
    assert (SOT_N_FFT, SOT_HOP) == (2048, 256)
    assert (TFW_N_FFT, TFW_HOP) == (1024, 512)
    assert SMOOTH_WINDOWS == (67, 127, 257, 509, 1021, 2053)
    assert SMOOTH_HOPS == (32, 63, 128, 254, 510, 1026)


def test_all_losses_are_zero_on_self_and_differentiable_after_shift() -> None:
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(7)
    target = torch.randn(1, 4096, generator=generator, dtype=torch.float64)
    shifted = torch.roll(target, 7, dims=-1).requires_grad_()
    objective = PaperObjectives(target)
    same = objective.values(target)
    values = objective.values(shifted)
    for index, name in enumerate(NAMES):
        assert float(same[name]) == 0.0
        assert float(values[name]) > 0.0
        (gradient,) = torch.autograd.grad(
            values[name].sum(), shifted, retain_graph=index < len(NAMES) - 1
        )
        assert bool(torch.isfinite(gradient).all())


def test_one_bound_target_broadcasts_over_candidate_rows() -> None:
    target = torch.randn(1, 4096, generator=torch.Generator().manual_seed(11), dtype=torch.float64)
    candidates = torch.stack((torch.roll(target[0], 3), torch.roll(target[0], 17)))
    objective = PaperObjectives(target)
    batched = objective.values(candidates)
    for name in NAMES:
        singles = torch.cat(
            [PaperObjectives(target, (name,))(row[None]) for row in candidates]
        )
        torch.testing.assert_close(batched[name], singles, rtol=1e-12, atol=1e-14)
