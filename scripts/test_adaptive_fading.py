"""Scheduled fading from infinity to 1 second / 1 octave, fresh fits only."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import torch
from _recovery_study_fit import decode_coordinates, encode_coordinates, fit
from icassp27_phrase.config import PAPER_OPTIMIZER, EventPhrase, initial_candidate
from icassp27_phrase.losses import CumulativeEnergyDistance
from icassp27_phrase.runtime import require_df2_backend
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.targets import load_target
from run_direction_recovery import describe
from test_fading_diagonal import fade_matrix
from test_fading_failures import CASES, frequency_matrix
from test_fading_failures import signature as earlier_signature
from test_octave_fading import octave_surfaces

ROOT = Path("results/adaptive-fading")
DOCS = Path("docs/adaptive-fading")
RAMP = 100
ANNEAL = 2000
MAXIMUM = 3000


def signature():
    return hashlib.sha256(earlier_signature().encode() + Path(__file__).read_bytes()).hexdigest()


class Objective:
    def __init__(self, audio):
        self.canonical = CumulativeEnergyDistance(audio)
        self.target_power = self.canonical._power(audio[None])[0].detach()
        self.mass = self.target_power.sum()
        nf, nt = self.target_power.shape
        self.hz = torch.arange(nf, dtype=torch.float64) * 4000 / 256
        span_f = math.log2(float(self.hz[-1]) / 20)
        span_t = (nt - 1) * 64 / 4000
        denominator = 10**0.005 - 1
        self.first = (9 * span_t / denominator, 9 * span_f / denominator)
        # Six finite stages, equally spaced in log horizon, from calibrated to 1.
        self.levels = [tuple(h ** (1 - i / 5) for h in self.first) for i in range(6)]

    def horizons(self, stage, alpha=1):
        if stage == 0 or (stage == 1 and alpha == 0):
            return None
        end = self.levels[stage - 1]
        if stage == 1:
            return tuple(h / alpha for h in end)
        start = self.levels[stage - 2]
        return tuple(
            math.exp((1 - alpha) * math.log(a) + alpha * math.log(b))
            for a, b in zip(start, end, strict=True)
        )

    def __call__(self, audio, stage, alpha=1):
        horizons = self.horizons(stage, alpha)
        if horizons is None:
            return self.canonical(audio)
        th, fh = horizons
        f = frequency_matrix(self.hz, fh)
        t = fade_matrix(self.target_power.shape[1], 64 / 4000, th)
        candidate = self.canonical._power(audio[None])[0]
        ref = (octave_surfaces(self.target_power, f, t) / self.mass).clamp_min(1e-12).sqrt()
        feature = (octave_surfaces(candidate, f, t) / self.mass).clamp_min(1e-12).sqrt()
        error = feature - ref
        return (
            torch.linalg.vector_norm(error.flatten(1), dim=1) / math.sqrt(error[0].numel())
        ).mean()


def schedule(step):
    if step == 0:
        return 0, 1.0
    if step <= RAMP:
        return 1, step / RAMP
    if step >= ANNEAL:
        return 6, 1.0
    position = 5 * (step - RAMP) / (ANNEAL - RAMP)
    return 2 + int(position), position % 1


def adaptive(audio, events, synth, obj, maximum=MAXIMUM, fading=True):
    initial = initial_candidate(events, device="cpu")
    raw = encode_coordinates(initial.f0_hz, initial.onset_seconds).requires_grad_()
    first, second = torch.zeros_like(raw), torch.zeros_like(raw)
    reference, scale = None, None
    patience, reductions, step = 0, 0, 0
    lr = 0.05
    best_value, stage_best = math.inf, math.inf
    stage_raw = raw.detach().clone()
    trajectory = []
    started = time.perf_counter()
    reason = "maximum updates"
    while True:
        stage, alpha = schedule(step) if fading else (0, 1.0)
        f0, onset = decode_coordinates(raw)
        rendered = synth.render_batch(f0[None], onset[None])[0]
        train = obj(rendered, stage, alpha)
        with torch.no_grad():
            canonical = float(obj.canonical(rendered))
        value = float(train.detach())
        assert math.isfinite(value) and math.isfinite(canonical)
        if scale is None:
            scale = value
            assert scale > 0
        if canonical < best_value:
            best_value = canonical
            best_raw = raw.detach().clone()
            best_update = step
        # Only compare changing-objective values once the final horizon is fixed.
        if step >= ANNEAL:
            if value < stage_best:
                stage_best = value
                stage_raw = raw.detach().clone()
            if reference is None or value <= reference * (1 - 1e-4):
                reference = value
                patience = 0
                reductions = 0
            else:
                patience += 1
        trajectory.append(
            dict(
                update=step,
                stage=stage,
                alpha=alpha,
                training_loss=value,
                canonical_loss=canonical,
                best_canonical_loss=best_value,
                patience=patience,
                learning_rate=lr,
                horizons=obj.horizons(stage, alpha),
                f0_hz=f0.detach().tolist(),
                onset_seconds=onset.detach().tolist(),
            )
        )
        if step % 500 == 0:
            print(
                "annealed" if fading else "unfaded",
                step,
                stage,
                alpha,
                canonical,
                patience,
                flush=True,
            )
        if step >= maximum:
            break
        if step >= ANNEAL:
            if patience >= 250:
                reason = "terminal patience"
                break
            if patience >= (reductions + 1) * 100:
                reductions += 1
                lr = max(lr * 0.3, 1e-5)
                raw = stage_raw.detach().clone().requires_grad_()
                first, second = torch.zeros_like(raw), torch.zeros_like(raw)
                continue
        (gradient,) = torch.autograd.grad(train / scale, raw)
        assert torch.isfinite(gradient).all()
        step += 1
        first = 0.9 * first + (1 - 0.9) * gradient
        second = 0.999 * second + (1 - 0.999) * gradient.square()
        raw = (
            (raw - lr * (first / (1 - 0.9**step)) / ((second / (1 - 0.999**step)).sqrt() + 1e-8))
            .detach()
            .requires_grad_()
        )
    bf, bt = decode_coordinates(best_raw)
    lf, lt = decode_coordinates(raw)
    return dict(
        best_phrase=EventPhrase(bf.detach(), bt.detach()),
        last_phrase=EventPhrase(lf.detach(), lt.detach()),
        updates=step,
        best_update=best_update,
        best_loss=best_value,
        trajectory=trajectory,
        stopped_by=reason,
        wall_seconds=time.perf_counter() - started,
        final_stage=stage,
        final_alpha=alpha,
        initial_scale=scale,
        annealing_updates=ANNEAL,
        fading=fading,
    )


def qualify():
    torch.manual_seed(794)
    synth = PhraseSynth()
    _, target = load_target(4, 6)
    audio = synth.render(target).detach()
    obj = Objective(audio)
    candidate = synth.render(initial_candidate(4, device="cpu")).detach().requires_grad_()
    assert obj(candidate, 0) == obj.canonical(candidate)
    assert obj(candidate, 1, 0) == obj.canonical(candidate)
    first = obj.horizons(1)
    f = frequency_matrix(obj.hz, first[1])
    t = fade_matrix(obj.target_power.shape[1], 0.016, first[0])
    assert float(f[-1, 0] * t[-1, 0]) >= 0.99
    for stage in range(2, 7):
        assert obj.horizons(stage, 0) == obj.horizons(stage - 1) or all(
            math.isclose(a, b, rel_tol=1e-14)
            for a, b in zip(obj.horizons(stage, 0), obj.horizons(stage - 1), strict=True)
        )
    assert obj.horizons(6) == (1, 1)
    from test_octave_fading import OctaveFading

    torch.testing.assert_close(obj(candidate, 6), OctaveFading(audio)(candidate), rtol=0, atol=0)
    for stage, alpha in ((0, 1), (1, 0.5), (3, 0.5), (6, 1)):
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
    control = fit(
        audio,
        4,
        "diagonal",
        synth=synth,
        objective=obj.canonical,
        config=replace(PAPER_OPTIMIZER, maximum_updates=12),
    )
    actual = adaptive(audio, 4, synth, obj, maximum=12, fading=False)
    # Same mathematical Adam update; tolerate rounding due to scalar arithmetic association.
    torch.testing.assert_close(
        actual["best_phrase"].f0_hz, control.best_phrase.f0_hz, rtol=1e-12, atol=1e-12
    )
    assert all(s["stage"] == 0 for s in actual["trajectory"])
    for step in range(ANNEAL + 1):
        stage, alpha = schedule(step)
        assert 0 <= stage <= 6 and 0 <= alpha <= 1
    assert obj.horizons(*schedule(0)) is None
    assert obj.horizons(*schedule(ANNEAL)) == (1.0, 1.0)

    class ConstantObjective:
        @staticmethod
        def canonical(value):
            return value.sum() * 0 + 1

        def __call__(self, value, stage, alpha):
            return self.canonical(value)

        def horizons(self, stage, alpha):
            return obj.horizons(stage, alpha)

    class ToySynth:
        @staticmethod
        def render_batch(frequency, onset):
            return frequency + onset

    plateau = adaptive(torch.ones(4), 4, ToySynth(), ConstantObjective(), maximum=2300)
    assert plateau["stopped_by"] == "terminal patience"
    assert plateau["final_stage"] == 6 and plateau["final_alpha"] == 1
    assert plateau["updates"] >= ANNEAL
    assert min(s["learning_rate"] for s in plateau["trajectory"]) == 0.05 * 0.3 * 0.3
    assert all(
        s["patience"] == 0 and s["learning_rate"] == 0.05
        for s in plateau["trajectory"]
        if s["update"] < ANNEAL
    )
    DOCS.mkdir(parents=True, exist_ok=True)
    report = dict(
        passed=True,
        signature=signature(),
        calibrated_horizons=obj.first,
        levels=obj.levels,
        maximum_first_stage_attenuation=1 - float(f[-1, 0] * t[-1, 0]),
        exact_initial_unfaded=True,
        exact_terminal_one_octave=True,
        gradient_checks=True,
        terminal_patience_and_rollback_checked=True,
    )
    (DOCS / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report, flush=True)


def run(index):
    assert 0 <= index < 14
    q = json.loads((DOCS / "qualification.json").read_text())
    assert q["passed"] and q["signature"] == signature()
    case = CASES[index // 2]
    variant = "annealed" if index % 2 else "unfaded"
    synth = PhraseSynth()
    meta, target = load_target(case["events"], case["target"])
    audio = synth.render(target).detach()
    obj = Objective(audio)
    result = adaptive(audio, case["events"], synth, obj, fading=variant == "annealed")
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
        levels=obj.levels,
        renderer=synth.provenance(),
        amplitudes="fixed .8",
        log_weighing=False,
        checkpoint_selection="strict best unfaded diagonal loss over entire fresh run",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    ROOT.mkdir(parents=True, exist_ok=True)
    with gzip.open(ROOT / f"{index:02d}.json.gz", "wt") as handle:
        json.dump(out, handle)
    print("COMPLETE", index, variant, out["metrics"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=int)
    p.add_argument("--qualify", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    require_df2_backend("cpu")
    qualify() if args.qualify else run(args.index)
