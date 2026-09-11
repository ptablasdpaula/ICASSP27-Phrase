"""Exercise the notebook's full CPU render/loss/gradient path in memory."""

from __future__ import annotations

import json
import time

import torch

from icassp27_phrase import (
    OptimizerConfig,
    PhraseSynth,
    configure_reproducibility,
    fit,
    load_target,
)


def main() -> None:
    configure_reproducibility()
    device = torch.device("cpu")
    synth = PhraseSynth().to(device)
    metadata, phrase = load_target(2, 1, device=device)
    started = time.perf_counter()
    with torch.no_grad():
        target = synth.render(phrase).detach()
    rendered_seconds = time.perf_counter() - started
    if target.shape != (synth.sample_count,) or not bool(torch.isfinite(target).all()):
        raise RuntimeError("CPU notebook qualification produced malformed target audio")

    result = fit(
        target,
        cardinality=metadata.cardinality,
        loss_name="BiCuL",
        synth=synth,
        config=OptimizerConfig(
            plateau_patience=1,
            stop_patience=2,
            maximum_updates=1,
        ),
    )
    if result.evaluations != 2 or result.updates != 1:
        raise RuntimeError("CPU notebook qualification did not complete one update")
    if not all(
        torch.isfinite(torch.tensor(value))
        for value in (result.initial_loss, result.best_loss, result.wall_seconds)
    ):
        raise FloatingPointError("CPU notebook qualification returned non-finite metrics")
    print(
        json.dumps(
            {
                "device": "cpu",
                "threads": torch.get_num_threads(),
                "cardinality": metadata.cardinality,
                "loss": result.loss_name,
                "target_render_seconds": rendered_seconds,
                "two_evaluations_seconds": result.wall_seconds,
                "initial_loss": result.initial_loss,
                "best_loss": result.best_loss,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
