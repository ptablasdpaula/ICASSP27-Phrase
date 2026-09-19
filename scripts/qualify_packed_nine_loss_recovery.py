"""Verify row independence and the packed execution plan on CUDA."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from icassp27_phrase.runtime import require_df2_backend
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.targets import load_target
from run_nine_loss_recovery import LOSSES, PairedObjective, signature
from run_packed_nine_loss_recovery import SHARD_SPECS, TOTAL_SHARDS, plan_hash


def qualify(device: str) -> None:
    require_df2_backend(device)
    synth = PhraseSynth().to(device)
    targets, candidates = [], []
    for number in (1, 2, 3):
        _, phrase = load_target(2, number, device=device)
        targets.append(synth.render(phrase).detach())
        scale = torch.tensor([0.91 + number * 0.01, 1.07 - number * 0.01], device=device)
        shift = torch.tensor([0.021 + number * 0.003, -0.017], device=device)
        candidates.append(
            synth(
                (phrase.f0_hz * scale).clamp(80, 320)[None],
                (phrase.onset_seconds + shift).clamp(0.2, 1.8)[None],
            )[0].detach()
        )
    target = torch.stack(targets)
    candidate = torch.stack(candidates)
    # Recovery drops completed rows and selects their corresponding cached
    # targets. Compare that path against a freshly bound objective at the same
    # batch shape, avoiding irrelevant batch-size-dependent cuFFT roundoff.
    indices = torch.tensor([0, 2], device=device)
    selected_target = target[indices]
    selected_candidate = candidate[indices]
    checks = []
    for loss in LOSSES:
        selected_audio = selected_candidate.clone().requires_grad_(True)
        selected = PairedObjective(target, loss)(selected_audio, indices)
        (selected_gradient,) = torch.autograd.grad(selected.sum(), selected_audio)
        fresh_audio = selected_candidate.clone().requires_grad_(True)
        fresh = PairedObjective(selected_target, loss)(fresh_audio)
        (fresh_gradient,) = torch.autograd.grad(fresh.sum(), fresh_audio)
        torch.testing.assert_close(selected, fresh, rtol=2e-9, atol=2e-11)
        torch.testing.assert_close(selected_gradient, fresh_gradient, rtol=2e-8, atol=2e-10)
        checks.append(
            {
                "loss": loss,
                "value_max_abs": float((selected - fresh).abs().max()),
                "gradient_max_abs": float((selected_gradient - fresh_gradient).abs().max()),
            }
        )
    covered = {
        (loss, cardinality, target)
        for loss, cardinality, begin, count in SHARD_SPECS
        for target in range(begin, begin + count)
    }
    if len(covered) != len(LOSSES) * 5 * 150:
        raise AssertionError("packed plan does not cover every target exactly once")
    sig, hashes = signature()
    payload = {
        "passed": True,
        "scientific_signature": sig,
        "source_hashes": hashes,
        "execution_plan_sha256": plan_hash(),
        "total_shards": TOTAL_SHARDS,
        "total_fits": len(covered),
        "device": device,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "checks": checks,
    }
    path = Path("results/phrase-recovery-16k") / f"qualification-{device}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    qualify(args.device)
