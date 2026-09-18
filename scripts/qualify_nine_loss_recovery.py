"""Qualify paired objectives and per-row plateau semantics before production."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from icassp27_phrase.gradient_assessment import SharedObjectives
from icassp27_phrase.losses import CumulativeEnergyDistance
from icassp27_phrase.runtime import require_df2_backend
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.targets import load_target
from run_nine_loss_recovery import (
    LOSSES,
    SCHEDULE,
    PairedObjective,
    append_lr_events,
    signature,
)
from test_fading_diagonal import fade_matrix, surfaces


def expected(name, target, candidate):
    if name == "dec_cel":
        base = CumulativeEnergyDistance(target)
        target_power = base._power(target[None])[0]
        candidate_power = base._power(candidate[None])[0]
        frequency = fade_matrix(target_power.shape[0], 4000 / 256, 1000).to(target.device)
        time = fade_matrix(target_power.shape[1], 64 / 4000, 1).to(target.device)
        mass = target_power.sum()
        reference = (surfaces(target_power, frequency, time) / mass).clamp_min(1e-12).sqrt()
        value = (surfaces(candidate_power, frequency, time) / mass).clamp_min(1e-12).sqrt()
        error = value - reference
        return (
            torch.linalg.vector_norm(error.flatten(1), dim=1).mean()
            / (error.shape[-2] * error.shape[-1]) ** 0.5
        )
    if name == "tlog_cel":
        return CumulativeEnergyDistance(
            target, directions=("right_up", "right_down"), log_weighing=True
        )(candidate)
    bank = SharedObjectives(target)
    mapping = {
        "single_stft": "single_stft",
        "linear_mss": "linear_mss",
        "smooth_mss": "smooth_mss",
        "linear_jtfot": "linear_jtfot",
        "log_jtfot": "log_jtfot",
        "cel": "bidirectional_cumulative_energy",
        "log_cel": "log_quadrature_bicul",
    }
    return bank.values(candidate[None])[mapping[name]][0]


def qualify(device):
    require_df2_backend(device)
    synth = PhraseSynth().to(device)
    targets, candidates = [], []
    for number in (1, 2):
        _, phrase = load_target(2, number, device=device)
        targets.append(synth.render(phrase).detach())
        # A deterministic, nonidentical candidate with finite gradients.
        f0 = (phrase.f0_hz * torch.tensor([0.91, 1.08], device=device)).clamp(80, 320)
        onset = (phrase.onset_seconds + torch.tensor([0.037, -0.029], device=device)).clamp(
            0.2, 1.8
        )
        candidates.append(synth(f0[None], onset[None])[0].detach())
    target = torch.stack(targets)
    candidate = torch.stack(candidates).requires_grad_()
    checks = []
    for name in LOSSES:
        actual = PairedObjective(target, name)(candidate)
        reference = torch.stack([expected(name, target[i], candidate[i]) for i in range(2)])
        torch.testing.assert_close(actual, reference, rtol=2e-9, atol=2e-11)
        (ga,) = torch.autograd.grad(actual.sum(), candidate, retain_graph=True)
        (gr,) = torch.autograd.grad(reference.sum(), candidate, retain_graph=True)
        torch.testing.assert_close(ga, gr, rtol=2e-8, atol=2e-10)

        # Completed rows are removed during a production fit.  Verify that
        # selecting a live row also selects the matching cached target state.
        subset_candidate = candidate[1:2].detach().requires_grad_(True)
        subset = PairedObjective(target, name)(
            subset_candidate, torch.tensor([1], device=target.device)
        )
        torch.testing.assert_close(subset, reference[1:2], rtol=2e-9, atol=2e-11)
        checks.append({"loss": name, "value_max_abs": float((actual - reference).abs().max())})

    # Confirm the chosen bad-epoch convention against PyTorch itself.
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.Adam([parameter], lr=SCHEDULE.initial_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULE.factor,
        patience=SCHEDULE.plateau_patience,
        threshold=SCHEDULE.threshold,
        threshold_mode="rel",
        min_lr=SCHEDULE.minimum_lr,
    )
    scheduler.step(1.0)
    for _ in range(SCHEDULE.plateau_patience):
        scheduler.step(1.0)
    assert optimizer.param_groups[0]["lr"] == SCHEDULE.initial_lr
    scheduler.step(1.0)
    assert optimizer.param_groups[0]["lr"] == SCHEDULE.initial_lr * SCHEDULE.factor

    # Exercise noncontiguous simultaneous reductions, the same compacted-row
    # case used after some phrases in a batch have already stopped.
    events = [[] for _ in range(6)]
    indices = torch.tensor([1, 5], device=device)
    old_lr = torch.tensor([0.05, 0.025], dtype=torch.float64, device=device)
    live_lr = torch.tensor([0.0, 0.025, 0.0, 0.0, 0.0, 0.0125], device=device)
    updates = torch.tensor([0, 201, 0, 0, 0, 402], device=device)
    best = torch.tensor([0.0, 0.4, 0.0, 0.0, 0.0, 0.2], device=device)
    append_lr_events(events, indices, old_lr, live_lr, updates, best)
    assert events[1][0]["old_lr"] == 0.05 and events[1][0]["update"] == 201
    assert events[5][0]["old_lr"] == 0.025 and events[5][0]["update"] == 402

    sig, hashes = signature()
    payload = {
        "passed": True,
        "signature": sig,
        "source_hashes": hashes,
        "device": device,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "checks": checks,
        "scheduler": {
            "implementation": "per-phrase vectorised torch ReduceLROnPlateau semantics",
            "bad_epoch_boundary_checked": True,
            **SCHEDULE.__dict__,
        },
    }
    path = Path("results/nine-loss-recovery") / f"qualification-{device}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    qualify(args.device)
