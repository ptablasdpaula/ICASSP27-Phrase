"""Shorten time fade from infinity to .25s; frequency accumulation stays unfaded."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import subprocess
from pathlib import Path

import torch
from icassp27_phrase.config import initial_candidate
from icassp27_phrase.losses import CumulativeEnergyDistance
from icassp27_phrase.runtime import require_df2_backend
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.targets import load_target
from run_direction_recovery import describe
from test_adaptive_fading import CASES, adaptive, schedule
from test_adaptive_fading import signature as earlier_signature
from test_fading_diagonal import FadingDiagonal, fade_matrix, surfaces

ROOT = Path("results/time-fading")
DOCS = Path("docs/time-fading")


def signature():
    return hashlib.sha256(earlier_signature().encode() + Path(__file__).read_bytes()).hexdigest()


class TimeObjective:
    def __init__(self, audio):
        self.canonical = CumulativeEnergyDistance(audio)
        self.target_power = self.canonical._power(audio[None])[0].detach()
        self.mass = self.target_power.sum()
        self.frequency = fade_matrix(self.target_power.shape[0], 4000 / 256, None)
        span = (self.target_power.shape[1] - 1) * 0.016
        self.first = 9 * span / (10**0.01 - 1)

    def horizons(self, stage, alpha=1):
        if stage == 0 or (stage == 1 and alpha == 0):
            return None, None
        if stage == 1:
            return self.first / alpha, None
        progress = (stage - 2 + alpha) / 5
        return self.first * (0.25 / self.first) ** progress, None

    def __call__(self, audio, stage, alpha=1):
        th, _ = self.horizons(stage, alpha)
        if th is None:
            return self.canonical(audio)
        time = fade_matrix(self.target_power.shape[1], 0.016, th)
        power = self.canonical._power(audio[None])[0]
        ref = (
            (surfaces(self.target_power, self.frequency, time) / self.mass).clamp_min(1e-12).sqrt()
        )
        feature = (surfaces(power, self.frequency, time) / self.mass).clamp_min(1e-12).sqrt()
        error = feature - ref
        return (
            torch.linalg.vector_norm(error.flatten(1), dim=1) / math.sqrt(error[0].numel())
        ).mean()


def qualify():
    torch.manual_seed(794)
    old = json.loads(Path("docs/adaptive-fading/qualification.json").read_text())
    assert old["passed"] and old["signature"] == earlier_signature()
    synth = PhraseSynth()
    _, target = load_target(4, 6)
    audio = synth.render(target).detach()
    obj = TimeObjective(audio)
    candidate = synth.render(initial_candidate(4, device="cpu")).detach().requires_grad_()
    previous = math.inf
    for step in range(2001):
        th, fh = obj.horizons(*schedule(step))
        assert fh is None
        current = math.inf if th is None else th
        assert current <= previous
        previous = current
    assert obj.horizons(*schedule(0)) == (None, None)
    assert math.isclose(obj.horizons(*schedule(2000))[0], 0.25, rel_tol=1e-14)
    assert float(fade_matrix(obj.target_power.shape[1], 0.016, obj.first)[-1, 0]) >= 0.99
    assert torch.equal(obj.frequency, torch.ones_like(obj.frequency).tril())
    for step in (0, 1000, 2000):
        stage, alpha = schedule(step)
        th, _ = obj.horizons(stage, alpha)
        ref = obj.canonical if th is None else FadingDiagonal(audio, th, None)
        torch.testing.assert_close(obj(candidate, stage, alpha), ref(candidate), rtol=0, atol=0)
        torch.testing.assert_close(
            torch.autograd.grad(obj(candidate, stage, alpha), candidate)[0],
            torch.autograd.grad(ref(candidate), candidate)[0],
            rtol=0,
            atol=0,
        )
        same = audio.clone().requires_grad_()
        value = obj(same, stage, alpha)
        assert value == 0 and torch.autograd.grad(value, same)[0].abs().max() == 0
        direction = torch.randn_like(candidate)
        direction /= direction.norm()
        analytic = (
            torch.autograd.grad(obj(candidate, stage, alpha), candidate)[0] * direction
        ).sum()
        numeric = (
            obj(candidate + 1e-5 * direction, stage, alpha)
            - obj(candidate - 1e-5 * direction, stage, alpha)
        ) / 2e-5
        torch.testing.assert_close(analytic, numeric, rtol=1e-4, atol=1e-9)
    DOCS.mkdir(parents=True, exist_ok=True)
    report = dict(
        passed=True,
        signature=signature(),
        no_frequency_fading_all_updates=True,
        calibrated_first_time_horizon=obj.first,
        endpoint_and_midpoint_values_gradients_exact=True,
        self_loss_and_gradient=True,
        audio_finite_difference=True,
        qualified_optimizer_reused=True,
        checkpoint_monitor="original unfaded diagonal loss",
        log_weighing=False,
    )
    (DOCS / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report, flush=True)


def run(index):
    assert 0 <= index < 14
    q = json.loads((DOCS / "qualification.json").read_text())
    assert q["passed"] and q["signature"] == signature()
    case = CASES[index // 2]
    variant = "time_annealed" if index % 2 else "unfaded"
    synth = PhraseSynth()
    meta, target = load_target(case["events"], case["target"])
    audio = synth.render(target).detach()
    obj = TimeObjective(audio)
    result = adaptive(audio, case["events"], synth, obj, fading=bool(index % 2))
    best = result.pop("best_phrase")
    last = result.pop("last_phrase")
    with torch.no_grad():
        assert abs(float(obj.canonical(synth.render(best))) - result["best_loss"]) < 1e-10
    out = dict(
        index=index,
        case=case,
        variant=variant,
        signature=signature(),
        fit=result,
        metrics=describe(best, synth, audio, meta),
        last_metrics=describe(last, synth, audio, meta),
        best_phrase=dict(f0_hz=best.f0_hz.tolist(), onset_seconds=best.onset_seconds.tolist()),
        last_phrase=dict(f0_hz=last.f0_hz.tolist(), onset_seconds=last.onset_seconds.tolist()),
        renderer=synth.provenance(),
        amplitudes="fixed .8",
        log_weighing=False,
        checkpoint_selection="strict best original unfaded diagonal loss",
        fresh_start=True,
        frequency_horizon_hz=None,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / f"{index:02d}.json.gz"
    temporary = path.with_suffix(".tmp.gz")
    with gzip.open(temporary, "wt") as f:
        json.dump(out, f, separators=(",", ":"))
    temporary.replace(path)
    print("COMPLETE", index, variant, out["metrics"], "LAST", out["last_metrics"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qualify", action="store_true")
    p.add_argument("--index", type=int)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    require_df2_backend("cpu")
    qualify() if args.qualify else run(args.index)
