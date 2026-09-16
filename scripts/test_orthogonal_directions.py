"""Axis-only CeL escape from the archived C04-T0005 diagonal plateau."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
from _amplitude_fit import decode_coordinates, encode_coordinates, fit
from icassp27_phrase.config import EventPhrase
from icassp27_phrase.losses import CumulativeEnergyDistance, reverse_cumsum
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.targets import load_target
from test_amplitude_recovery import metrics

ROOT = Path("docs/orthogonal-escape-pilot")
STEPS = 1600
PERIOD = 800


def orthogonal_weight(step, variant):
    if variant == "eight_directions":
        return 0.5
    phase = (step % PERIOD) / PERIOD
    return 1.0 - abs(2.0 * phase - 1.0)


def surfaces(power):
    """[frequency,time] -> up, down, right, left; never sum the other axis."""
    return torch.stack((power.cumsum(-2), reverse_cumsum(power, -2),
                        power.cumsum(-1), reverse_cumsum(power, -1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=["all_four", "frequency_only", "diagonal_control",
                                                    "interpolating", "eight_directions"])
    args = parser.parse_args()
    assert [orthogonal_weight(s, "interpolating") for s in (0, 200, 400, 600, 800)] == [
        0.0, 0.5, 1.0, 0.5, 0.0]
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / f"{args.variant}.json"
    if path.exists():
        raise FileExistsError(path)
    # Verify axis isolation and inclusive reverse sums with an asymmetric toy grid.
    toy = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=torch.float64)
    expected = torch.tensor([[[1, 2, 3], [5, 7, 9]], [[5, 7, 9], [4, 5, 6]],
                             [[1, 3, 6], [4, 9, 15]], [[6, 5, 3], [15, 11, 6]]])
    assert torch.equal(surfaces(toy), expected)
    baseline = json.loads(Path("docs/amplitude-pilot/baseline.json").read_text())
    initial = EventPhrase(*(torch.tensor(baseline["best_phrase"][key], dtype=torch.float64)
                            for key in ["f0_hz", "onset_seconds"]))
    metadata, target = load_target(4, 6)
    synth = PhraseSynth()
    with torch.no_grad():
        audio = synth.render(target)
    canonical = CumulativeEnergyDistance(audio)
    target_power = canonical._power(audio[None])[0]
    mass = target_power.sum()
    reference = (surfaces(target_power) / mass).clamp_min(1e-12).sqrt().detach()

    def axis_terms(candidate):
        power = canonical._power(candidate[None])[0]
        error = (surfaces(power) / mass).clamp_min(1e-12).sqrt() - reference
        return torch.linalg.vector_norm(error.flatten(1), dim=1) / math.sqrt(power.numel())

    assert float(axis_terms(audio).max()) == 0
    raw = encode_coordinates(initial.f0_hz, initial.onset_seconds).requires_grad_()
    opt = torch.optim.Adam([raw], lr=0.05)
    best = float("inf")
    best_raw = None
    trajectory = []
    for step in range(STEPS + 1):
        f, t = decode_coordinates(raw)
        candidate = synth.render_batch(f[None], t[None])[0]
        diagonal = canonical(candidate)
        terms = axis_terms(candidate)
        objective = (diagonal if args.variant == "diagonal_control" else
                     terms[:2].mean() if args.variant == "frequency_only" else terms.mean())
        mixing = args.variant in ("interpolating", "eight_directions")
        weight = orthogonal_weight(step, args.variant) if mixing else None
        if mixing:
            objective = (1 - weight) * diagonal + weight * terms.mean()
        value = float(diagonal.detach())
        if value < best:
            best, best_raw = value, raw.detach().clone()
        if step == 0:
            assert abs(value - baseline["best_loss"]) < 1e-12
            scale = float(diagonal.detach()) if mixing else float(objective.detach())
        row = dict(update=step, canonical_loss=value, best_canonical_loss=best,
                   objective_loss=float(objective.detach()), axis_terms=terms.detach().tolist(),
                   f0_hz=f.detach().tolist(), onset_seconds=t.detach().tolist(),
                   metrics=metrics(f.detach().numpy(), t.detach().numpy(), metadata))
        if mixing:
            row["orthogonal_weight"] = weight
            reconstructed = (1 - weight) * value + weight * sum(row["axis_terms"]) / 4
            assert abs(row["objective_loss"] - reconstructed) < 1e-12
        if step % 100 == 0:
            gradient = torch.autograd.grad(objective, raw, retain_graph=True)[0]
            row["gradient_norm"] = float(gradient.norm())
            row["pitch_gradient_norm"] = float(gradient[:, 0].norm())
            row["onset_gradient_norm"] = float(gradient[:, 1].norm())
            print(args.variant, step, best, row["metrics"], flush=True)
        trajectory.append(row)
        if step == STEPS:
            break
        opt.zero_grad()
        (objective / scale).backward()
        assert bool(torch.isfinite(raw.grad).all())
        opt.step()
    f, t = decode_coordinates(best_raw)
    polish = fit(audio, 4, "bidirectional_cumulative_energy", synth=synth,
                 initial=EventPhrase(f.detach(), t.detach()), free_amplitudes=False)
    out = asdict(polish)
    out["best_phrase"] = dict(f0_hz=polish.best_phrase.f0_hz.tolist(),
                              onset_seconds=polish.best_phrase.onset_seconds.tolist())
    out["metrics"] = metrics(polish.best_phrase.f0_hz.numpy(),
                              polish.best_phrase.onset_seconds.numpy(), metadata)
    result = dict(variant=args.variant, target=asdict(metadata), renderer=synth.provenance(),
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                         text=True).strip(),
                  initial_objective_scale=scale, target_mass=float(mass),
                  interpolation_period=PERIOD if args.variant == "interpolating" else None,
                  exploration_trajectory=trajectory, best_exploration_loss=best, polish=out)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print("COMPLETE", args.variant, polish.best_loss, out["metrics"], flush=True)


if __name__ == "__main__":
    main()
