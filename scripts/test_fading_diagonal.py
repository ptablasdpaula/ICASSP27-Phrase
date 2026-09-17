"""Logarithmically fading diagonal accumulation on frozen eight-event failure."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
from _recovery_study_fit import fit
from icassp27_phrase.config import PAPER_OPTIMIZER, EventPhrase, initial_candidate
from icassp27_phrase.losses import CumulativeEnergyDistance
from icassp27_phrase.runtime import require_df2_backend
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.targets import load_target
from run_direction_recovery import describe, serialise_fit
from run_direction_recovery import signature as shared_signature

ROOT = Path("results/fading-diagonal")
DOCS = Path("docs/fading-diagonal")
BASE = Path("results/takeover-validation/raw/C08-T0000/baseline.json")
# Joint horizons in seconds and Hz, respectively. None means exact ordinary CeL.
SETTINGS = ((None, None), (2.0, 2000.0), (1.0, 1000.0), (0.5, 500.0), (0.25, 250.0))


def signature():
    h = hashlib.sha256(shared_signature().encode())
    h.update(Path(__file__).read_bytes())
    h.update(BASE.read_bytes())
    return h.hexdigest()


def fade_matrix(size, spacing, horizon):
    """Causal weights: max(0, 1-log(1+9*d/H)/log(10)); unit self-weight."""
    index = torch.arange(size, dtype=torch.float64)
    distance = (index[:, None] - index[None, :]) * spacing
    causal = distance >= 0
    if horizon is None:
        return causal.to(torch.float64)
    return (1 - torch.log1p(9 * distance.clamp_min(0) / horizon) / math.log(10)).clamp_min(
        0
    ) * causal


def surfaces(power, frequency, time):
    """Inclusive separable causal convolution from each of the four corners."""
    result = []
    for tr in (False, True):
        for fr in (False, True):
            axes = tuple(axis for axis, reverse in ((-2, fr), (-1, tr)) if reverse)
            oriented = power.flip(axes) if axes else power
            value = frequency @ oriented @ time.T
            result.append(value.flip(axes) if axes else value)
    return torch.stack(result)


class FadingDiagonal:
    def __init__(self, target, time_horizon, frequency_horizon):
        self.base = CumulativeEnergyDistance(target)
        self.ordinary = time_horizon is None
        power = self.base._power(target[None])[0]
        self.mass = power.sum().detach()
        self.frequency = fade_matrix(power.shape[0], 4000 / 256, frequency_horizon)
        self.time = fade_matrix(power.shape[1], 64 / 4000, time_horizon)
        self.reference = self.features(power).detach()

    def features(self, power):
        value = surfaces(power, self.frequency, self.time)
        return (value / self.mass).clamp_min(1e-12).sqrt()

    def __call__(self, audio):
        if self.ordinary:
            # Preserve the exact legacy arithmetic for the matched control.
            return self.base(audio)
        error = self.features(self.base._power(audio[None])[0]) - self.reference
        return (
            torch.linalg.vector_norm(error.flatten(1), dim=1) / math.sqrt(error[0].numel())
        ).mean()


def qualify():
    torch.manual_seed(794)
    p = torch.rand(5, 7, dtype=torch.float64, requires_grad=True)
    f, t = fade_matrix(5, 1, 4), fade_matrix(7, 1, 6)
    actual = surfaces(p, f, t)
    expected = torch.zeros_like(actual)
    # Independent nested rectangle sums, including all reverse orientations.
    for q, (tr, fr) in enumerate((tr, fr) for tr in (False, True) for fr in (False, True)):
        for k in range(5):
            for n in range(7):
                for i in range(5):
                    for j in range(7):
                        df, dt = (i - k if fr else k - i), (j - n if tr else n - j)
                        if df >= 0 and dt >= 0:
                            wf = max(0, 1 - math.log1p(9 * df / 4) / math.log(10))
                            wt = max(0, 1 - math.log1p(9 * dt / 6) / math.log(10))
                            expected[q, k, n] += p[i, j] * wf * wt
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)
    assert torch.autograd.gradcheck(lambda x: surfaces(x, f, t), (p,), fast_mode=True)
    ordinary = surfaces(p, fade_matrix(5, 1, None), fade_matrix(7, 1, None))
    for q, (tr, fr) in enumerate((tr, fr) for tr in (False, True) for fr in (False, True)):
        axes = tuple(axis for axis, rev in ((0, fr), (1, tr)) if rev)
        x = p.flip(axes) if axes else p
        x = x.cumsum(1).cumsum(0)
        torch.testing.assert_close(ordinary[q], x.flip(axes) if axes else x)
    synth = PhraseSynth()
    _, target = load_target(8, 1)
    audio = synth.render(target).detach()
    candidate = synth.render(initial_candidate(8, device="cpu")).detach().requires_grad_()
    for th, fh in SETTINGS:
        objective = FadingDiagonal(audio, th, fh)
        same = audio.clone().requires_grad_()
        loss = objective(same)
        assert loss == 0 and torch.autograd.grad(loss, same)[0].abs().max() == 0
        gradient = torch.autograd.grad(objective(candidate), candidate)[0]
        assert torch.isfinite(gradient).all() and gradient.norm() > 0
    # Audio gradient finite difference away from the zero-error norm singularity.
    objective = FadingDiagonal(audio, 1.0, 1000.0)
    direction = torch.randn_like(candidate)
    direction /= direction.norm()
    analytic = (torch.autograd.grad(objective(candidate), candidate)[0] * direction).sum()
    step = 1e-5
    numeric = (
        objective(candidate + step * direction) - objective(candidate - step * direction)
    ) / (2 * step)
    torch.testing.assert_close(analytic, numeric, rtol=1e-4, atol=1e-9)
    DOCS.mkdir(parents=True, exist_ok=True)
    report = dict(
        passed=True,
        signature=signature(),
        explicit_rectangles=True,
        reverse_directions=True,
        no_fade_cumsum_limit=True,
        autograd_and_audio_finite_difference=True,
        zero_self_loss_and_gradient=True,
    )
    (DOCS / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def run(index):
    assert 0 <= index < 10
    q = json.loads((DOCS / "qualification.json").read_text())
    assert q["passed"] and q["signature"] == signature()
    mode = "plateau" if index < 5 else "fresh"
    th, fh = SETTINGS[index % 5]
    base = json.loads(BASE.read_text())
    initial = (
        initial_candidate(8, device="cpu")
        if mode == "fresh"
        else EventPhrase(
            *(
                torch.tensor(base["best_phrase"][k], dtype=torch.float64)
                for k in ("f0_hz", "onset_seconds")
            )
        )
    )
    synth = PhraseSynth()
    meta, target = load_target(8, 1)
    audio = synth.render(target).detach()
    objective = FadingDiagonal(audio, th, fh)
    initial_metrics = describe(initial, synth, audio, meta)

    def progress(snapshot, _audio):
        if snapshot.update % 250 == 0:
            print(index, mode, th, snapshot.update, snapshot.best_loss, flush=True)

    result = fit(
        audio,
        8,
        "fading_diagonal",
        synth=synth,
        objective=objective,
        initial=initial,
        progress=progress,
    )
    best = result.best_phrase
    canonical = CumulativeEnergyDistance(audio)
    with torch.no_grad():
        assert abs(float(objective(synth.render(best))) - result.best_loss) < 1e-10
        initial_canonical = float(canonical(synth.render(initial)))
        final_canonical = float(canonical(synth.render(best)))
    output = dict(
        index=index,
        mode=mode,
        target=asdict(meta),
        time_horizon_seconds=th,
        frequency_horizon_hz=fh,
        signature=signature(),
        optimizer=asdict(PAPER_OPTIMIZER),
        renderer=synth.provenance(),
        initial_metrics=initial_metrics,
        metrics=describe(best, synth, audio, meta),
        fit=serialise_fit(result),
        initial_canonical_loss=initial_canonical,
        final_canonical_loss=final_canonical,
        selection="strict best own training objective; no parameter-error selection",
        amplitudes="fixed 0.8",
        log_weighing=False,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / f"{index:02d}.json.gz"
    with gzip.open(path.with_suffix(".tmp.gz"), "wt") as handle:
        json.dump(output, handle)
    path.with_suffix(".tmp.gz").replace(path)
    print("COMPLETE", index, output["metrics"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int)
    parser.add_argument("--qualify", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    require_df2_backend("cpu")
    if args.qualify:
        qualify()
    else:
        run(args.index)
