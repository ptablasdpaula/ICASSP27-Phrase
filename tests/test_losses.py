from __future__ import annotations

import pytest
import torch
from icassp27_phrase.losses import (
    FABIANI_TIME_SCALE_HZ_PER_SECOND,
    LOSS_LABELS,
    build_loss,
    canonical_loss_name,
)


def test_paper_loss_labels_resolve_to_eight_distinct_losses() -> None:
    assert LOSS_LABELS == (
        "L_1", "L_2", "MSS", "SOT", "TFW_2", "TFW_2 (1s=1oct)",
        "BiCuL", "LogQ_BiCuL",
    )
    assert len({canonical_loss_name(name) for name in LOSS_LABELS}) == 8
    assert FABIANI_TIME_SCALE_HZ_PER_SECOND == 1_000.0


@pytest.mark.parametrize("name", LOSS_LABELS)
def test_each_loss_is_zero_on_self_and_positive_after_shift(name: str) -> None:
    generator = torch.Generator().manual_seed(7)
    target = torch.randn(8_000, generator=generator, dtype=torch.float64)
    metric = build_loss(name, target)
    values = metric(torch.stack((target, torch.roll(target, 7))))
    assert float(values[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(values[1]) > 0.0
    assert bool(torch.isfinite(values).all())
