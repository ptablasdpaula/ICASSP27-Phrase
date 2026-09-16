"""Plateau-triggered smooth direction rotation versus a matched uniform control."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
from _amplitude_fit import decode_coordinates, encode_coordinates, fit
from icassp27_phrase.config import EventPhrase
from icassp27_phrase.losses import CumulativeEnergyDistance
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.targets import load_target
from test_amplitude_recovery import metrics

ROOT = Path("docs/swap-escape-pilot/rotation")
STEPS = 1600


def weights(step, rotating):
    if not rotating:
        return torch.full((4,), 0.25, dtype=torch.float64)
    # Clockwise corner order: right/up, right/down, left/down, left/up.
    angles = torch.tensor([0, math.pi / 2, 3 * math.pi / 2, math.pi], dtype=torch.float64)
    theta = 2 * math.pi * step / 800
    preference = torch.softmax(4 * torch.cos(theta - angles), dim=0)
    strength = 0.9 * min(step / 100, 1)
    return (1 - strength) / 4 + strength * preference


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    ROOT.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(Path("docs/amplitude-pilot/baseline.json").read_text())
    initial = EventPhrase(
        *(
            torch.tensor(baseline["best_phrase"][key], dtype=torch.float64)
            for key in ["f0_hz", "onset_seconds"]
        )
    )
    metadata, target = load_target(4, 6)
    synth = PhraseSynth()
    with torch.no_grad():
        audio = synth.render(target)
    objective = CumulativeEnergyDistance(audio)
    provenance = dict(
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        steps=STEPS,
        period=800,
        ramp=100,
        concentration=4,
        strength=0.9,
        lr=0.05,
        amplitudes="fixed .8",
        times="independent bounded logits",
        selection="strict-best original uniform CeL, then canonical polish",
        control="same 1600 updates and Adam, but constant uniform weights",
        stopping="fixed exploration budget, not changing weighted loss or gradient norm",
    )
    (ROOT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for name, rotating in [("rotating", True), ("uniform_control", False)]:
        path = ROOT / f"{name}.json"
        if path.exists():
            raise FileExistsError(path)
        raw = encode_coordinates(initial.f0_hz, initial.onset_seconds).requires_grad_()
        opt = torch.optim.Adam([raw], lr=0.05, betas=(0.9, 0.999), eps=1e-8)
        best = float("inf")
        best_raw = None
        trajectory = []
        for step in range(STEPS + 1):
            f, t = decode_coordinates(raw)
            terms = objective.directional_distances(synth.render_batch(f[None], t[None])[0])[0]
            uniform = terms.mean()
            w = weights(step, rotating)
            weighted = (w * terms).sum()
            assert bool(torch.isfinite(terms).all())
            value = float(uniform.detach())
            if value < best:
                best = value
                best_raw = raw.detach().clone()
            if step == 0:
                assert abs(value - baseline["best_loss"]) < 1e-12
            row = dict(
                update=step,
                canonical_loss=value,
                best_canonical_loss=best,
                weighted_loss=float(weighted.detach()),
                weights=w.tolist(),
                directional_losses=terms.detach().tolist(),
                f0_hz=f.detach().tolist(),
                onset_seconds=t.detach().tolist(),
                metrics=metrics(f.detach().numpy(), t.detach().numpy(), metadata),
            )
            if step % 100 == 0:
                gs = torch.stack(
                    [torch.autograd.grad(term, raw, retain_graph=True)[0] for term in terms]
                )
                row["directional_gradient_norms"] = gs.flatten(1).norm(dim=1).tolist()
                row["uniform_gradient_norm"] = float(gs.mean(0).norm())
                row["weighted_gradient_norm"] = float((gs * w[:, None, None]).sum(0).norm())
                print(name, step, best, row["metrics"], flush=True)
            trajectory.append(row)
            if step == STEPS:
                break
            opt.zero_grad()
            (weighted / baseline["best_loss"]).backward()
            assert bool(torch.isfinite(raw.grad).all())
            opt.step()
        f, t = decode_coordinates(best_raw)
        polish = fit(
            audio,
            4,
            "bidirectional_cumulative_energy",
            synth=synth,
            initial=EventPhrase(f.detach(), t.detach()),
            free_amplitudes=False,
        )
        out = asdict(polish)
        out["best_phrase"] = dict(
            f0_hz=polish.best_phrase.f0_hz.tolist(),
            onset_seconds=polish.best_phrase.onset_seconds.tolist(),
        )
        out["metrics"] = metrics(
            polish.best_phrase.f0_hz.numpy(), polish.best_phrase.onset_seconds.numpy(), metadata
        )
        result = dict(exploration_trajectory=trajectory, best_exploration_loss=best, polish=out)
        path.write_text(json.dumps(result, indent=2) + "\n")
        print("COMPLETE", name, polish.best_loss, out["metrics"], flush=True)


if __name__ == "__main__":
    main()
