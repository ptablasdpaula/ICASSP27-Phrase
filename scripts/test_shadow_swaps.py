"""Isolated lookahead probes: no trial parameter or Adam-state update reaches the real fit."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch
from _amplitude_fit import decode_coordinates, encode_coordinates
from icassp27_phrase.losses import build_loss
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.targets import load_target

ROOT = Path("docs/shadow-swap-pilot")


@dataclass
class State:
    raw: torch.Tensor
    first: torch.Tensor
    second: torch.Tensor
    step: int = 0

    def clone(self):
        return State(self.raw.detach().clone(), self.first.clone(), self.second.clone(), self.step)


def advance(state, permutation, objective, synth, scale):
    raw = state.raw.detach().clone().requires_grad_()
    f, t = decode_coordinates(raw)
    loss = objective(synth.render_batch(f[permutation][None], t[None])[0])
    (g,) = torch.autograd.grad(loss / scale, raw)
    assert bool(torch.isfinite(g).all())
    first = 0.9 * state.first + 0.1 * g
    second = 0.999 * state.second + 0.001 * g.square()
    step = state.step + 1
    updated = raw - 0.05 * (first / (1 - 0.9**step)) / ((second / (1 - 0.999**step)).sqrt() + 1e-8)
    return State(updated.detach(), first.detach(), second.detach(), step), float(loss.detach())


def score(state, permutation, objective, synth):
    with torch.no_grad():
        f, t = decode_coordinates(state.raw)
        return float(objective(synth.render_batch(f[permutation][None], t[None])[0]))


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / "probes.json"
    if path.exists():
        raise FileExistsError(path)
    baseline = json.loads(Path("docs/amplitude-pilot/baseline.json").read_text())
    f, t = [
        torch.tensor(baseline["best_phrase"][key], dtype=torch.float64)
        for key in ["f0_hz", "onset_seconds"]
    ]
    raw = encode_coordinates(f, t)
    real = State(raw, torch.zeros_like(raw), torch.zeros_like(raw))
    synth = PhraseSynth()
    _, target = load_target(4, 6)
    with torch.no_grad():
        audio = synth.render(target)
    objective = build_loss("bidirectional_cumulative_energy", audio)
    scale = baseline["best_loss"]
    identity = torch.arange(4)
    records = []
    # Check both fresh plateau restart and the live state after one ordinary update.
    for starting_point in ["stuck", "after_real_step"]:
        if starting_point == "after_real_step":
            real, _ = advance(real, identity, objective, synth, scale)
        before = real.clone()
        for pair in [None, (0, 1), (1, 2), (2, 3)]:
            permutation = identity.clone()
            if pair is not None:
                i, j = pair
                permutation[i], permutation[j] = identity[j], identity[i]
            trial = real.clone()
            initial = score(trial, permutation, objective, synth)
            for depth in range(1, 21):
                trial, _ = advance(trial, permutation, objective, synth, scale)
                if depth in [1, 5, 20]:
                    records.append(
                        dict(
                            start=starting_point,
                            pair=pair,
                            depth=depth,
                            initial=initial,
                            after=score(trial, permutation, objective, synth),
                        )
                    )
            # Trial backward must not touch live parameters, gradients or moments.
            for name in ["raw", "first", "second"]:
                assert torch.equal(getattr(real, name), getattr(before, name))
                assert getattr(real, name).grad is None
            assert real.step == before.step
    for row in records:
        keep = next(
            r
            for r in records
            if r["start"] == row["start"] and r["depth"] == row["depth"] and r["pair"] is None
        )
        row["own_trial_improvement"] = row["initial"] - row["after"]
        row["advantage_over_unswapped_trial"] = keep["after"] - row["after"]
        row["advantage_over_real_before_trial"] = keep["initial"] - row["after"]
        # Two-way softmax over complete bijections, diagnostic temperature only.
        row["swap_probability_vs_keep"] = float(
            torch.softmax(
                torch.tensor([-keep["after"] / scale, -row["after"] / scale], dtype=torch.float64),
                0,
            )[1]
        )
    out = dict(
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        fixed_amplitudes=0.8,
        loss="four-direction unweighted CeL",
        padding=2.048,
        normalization="fixed real baseline initial loss for both trial branches",
        state_isolation_checks="passed exact equality; no live .grad populated",
        records=records,
    )
    path.write_text(json.dumps(out, indent=2) + "\n")
    for row in records:
        if row["start"] == "stuck" and row["depth"] in [1, 20]:
            print(row, flush=True)


if __name__ == "__main__":
    main()
