"""Numerically qualify the public CUDA renderer without writing artefacts."""

from __future__ import annotations

import json

import torch
from torch.profiler import ProfilerActivity, profile

from icassp27_phrase import (
    PhraseSynth,
    build_loss,
    configure_reproducibility,
    fit,
    load_target,
)
from icassp27_phrase.config import OptimizerConfig, initial_candidate
from icassp27_phrase.runtime import require_accelerated_backend


def main() -> None:
    configure_reproducibility()
    device = torch.device("cuda")
    require_accelerated_backend(device)
    synth = PhraseSynth().to(device)
    _, target = load_target(4, 1, device=device)
    initial = initial_candidate(4, device=device)
    candidate_f0 = initial.f0_hz.detach().clone().requires_grad_(True)
    candidate_onset = initial.onset_seconds.detach().clone().requires_grad_(True)

    with torch.no_grad():
        target_audio = synth.render(target)
    metric = build_loss("BiCuL", target_audio)
    with profile(activities=[ProfilerActivity.CPU]) as trace:
        candidate_audio = synth.render_batch(candidate_f0[None], candidate_onset[None])[0]
        loss = metric(candidate_audio)
        gradient = torch.autograd.grad(loss, (candidate_f0, candidate_onset))
        torch.cuda.synchronize(device)

    values = (target_audio, candidate_audio, loss, *gradient)
    if not all(bool(torch.isfinite(value.detach()).all()) for value in values):
        raise FloatingPointError("GPU qualification produced a non-finite value")
    if any(float(value.detach().abs().sum()) == 0.0 for value in gradient):
        raise RuntimeError("GPU qualification produced a zero control gradient")
    operators = {event.key for event in trace.key_averages()}
    if "torchlpc::lpc" not in operators:
        raise RuntimeError("the render did not execute the registered torchlpc::lpc operator")
    smoke_fit = fit(
        target_audio,
        cardinality=4,
        loss_name="BiCuL",
        synth=synth,
        config=OptimizerConfig(maximum_updates=1),
    )
    if smoke_fit.updates != 1 or smoke_fit.evaluations != 2:
        raise RuntimeError("the optimizer smoke fit violated its evaluation contract")

    print(
        json.dumps(
            {
                "status": "pass",
                "device": torch.cuda.get_device_name(device),
                "capability": ".".join(map(str, torch.cuda.get_device_capability(device))),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "dtype": str(candidate_audio.dtype),
                "samples": candidate_audio.numel(),
                "loss": float(loss.detach()),
                "maximum_pitch_gradient": float(gradient[0].detach().abs().max()),
                "maximum_onset_gradient": float(gradient[1].detach().abs().max()),
                "observed_operator": "torchlpc::lpc",
                "optimizer_smoke_evaluations": smoke_fit.evaluations,
                "optimizer_smoke_updates": smoke_fit.updates,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
