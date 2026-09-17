"""One-second / one-octave logarithmic fade on C08-T0000, two initial states."""

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
from test_fading_diagonal import BASE, FadingDiagonal, fade_matrix
from test_fading_diagonal import signature as previous_signature

ROOT = Path("results/octave-fading")
DOCS = Path("docs/octave-fading")


def signature():
    return hashlib.sha256(previous_signature().encode() + Path(__file__).read_bytes()).hexdigest()


def frequency_matrix(frequencies):
    coordinates = torch.log2(frequencies.clamp_min(20) / 20)
    distance = (coordinates[:, None] - coordinates[None, :]).abs()
    weights = (1 - torch.log1p(9 * distance) / math.log(10)).clamp_min(0)
    # Causality is by bin index, including the distinct bins below the 20-Hz floor.
    return weights.tril()


def octave_surfaces(power, frequency, time):
    # On a nonuniform octave grid the reverse kernel is the transpose, NOT a
    # reflection of the forward kernel as on the original uniform-Hz grid.
    return torch.stack(
        [
            (frequency.T if fr else frequency) @ power @ (time if tr else time.T)
            for tr in (False, True)
            for fr in (False, True)
        ]
    )


class OctaveFading(FadingDiagonal):
    def __init__(self, target):
        self.base = CumulativeEnergyDistance(target)
        self.ordinary = False
        power = self.base._power(target[None])[0]
        self.mass = power.sum().detach()
        self.frequency = frequency_matrix(
            torch.arange(power.shape[0], dtype=torch.float64) * 4000 / 256
        )
        self.time = fade_matrix(power.shape[1], 64 / 4000, 1.0)
        self.reference = self.features(power).detach()

    def features(self, power):
        return (
            (octave_surfaces(power, self.frequency, self.time) / self.mass).clamp_min(1e-12).sqrt()
        )


def qualify():
    torch.manual_seed(794)
    hz = torch.tensor([0, 15.625, 31.25, 62.5, 93.75], dtype=torch.float64)
    f, t = frequency_matrix(hz), fade_matrix(4, 0.25, 1)
    p = torch.rand(5, 4, dtype=torch.float64, requires_grad=True)
    expected = torch.zeros(4, 5, 4, dtype=torch.float64)
    for q, (tr, fr) in enumerate((tr, fr) for tr in (False, True) for fr in (False, True)):
        for k in range(5):
            for n in range(4):
                for i in range(5):
                    for j in range(4):
                        if (i >= k if fr else i <= k) and (j >= n if tr else j <= n):
                            distance = abs(math.log2(max(float(hz[k]), 20) / max(float(hz[i]), 20)))
                            wf = max(0, 1 - math.log1p(9 * distance) / math.log(10))
                            wt = max(0, 1 - math.log1p(9 * abs(n - j) * 0.25) / math.log(10))
                            expected[q, k, n] += p[i, j] * wf * wt
    torch.testing.assert_close(octave_surfaces(p, f, t), expected, atol=1e-14, rtol=1e-14)
    assert torch.autograd.gradcheck(lambda x: octave_surfaces(x, f, t), (p,), fast_mode=True)
    assert abs(float(f[3, 2])) < 1e-15  # Exactly one octave has zero retained power.
    synth = PhraseSynth()
    _, target = load_target(8, 1)
    audio = synth.render(target).detach()
    obj = OctaveFading(audio)
    same = audio.clone().requires_grad_()
    value = obj(same)
    assert value == 0 and torch.autograd.grad(value, same)[0].abs().max() == 0
    candidate = synth.render(initial_candidate(8, device="cpu")).detach().requires_grad_()
    direction = torch.randn_like(candidate)
    direction /= direction.norm()
    analytic = (torch.autograd.grad(obj(candidate), candidate)[0] * direction).sum()
    numeric = (obj(candidate + 1e-5 * direction) - obj(candidate - 1e-5 * direction)) / 2e-5
    torch.testing.assert_close(analytic, numeric, rtol=1e-4, atol=1e-9)
    DOCS.mkdir(parents=True, exist_ok=True)
    report = dict(
        passed=True,
        signature=signature(),
        explicit_all_four_directions=True,
        nonuniform_reverse_kernel=True,
        octave_cutoff=True,
        self_loss_gradient=True,
        autograd_and_audio_finite_difference=True,
    )
    (DOCS / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report, flush=True)


def run(index):
    assert index in (0, 1)
    q = json.loads((DOCS / "qualification.json").read_text())
    assert q["passed"] and q["signature"] == signature()
    mode = "plateau" if index == 0 else "fresh"
    base = json.loads(BASE.read_text())
    initial = (
        initial_candidate(8, device="cpu")
        if index
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
    obj = OctaveFading(audio)

    def progress(snapshot, _):
        if snapshot.update % 250 == 0:
            print(mode, snapshot.update, snapshot.best_loss, flush=True)

    result = fit(
        audio, 8, "octave_fading", synth=synth, objective=obj, initial=initial, progress=progress
    )
    with torch.no_grad():
        rendered = synth.render(result.best_phrase)
        assert abs(float(obj(rendered)) - result.best_loss) < 1e-10
        canonical = float(obj.base(rendered))
    output = dict(
        mode=mode,
        target=asdict(meta),
        signature=signature(),
        fit=serialise_fit(result),
        metrics=describe(result.best_phrase, synth, audio, meta),
        initial_metrics=describe(initial, synth, audio, meta),
        canonical_loss=canonical,
        renderer=synth.provenance(),
        optimizer=asdict(PAPER_OPTIMIZER),
        time_horizon_seconds=1,
        frequency_horizon_octaves=1,
        frequency_floor_hz=20,
        log_weighing=False,
        amplitudes="fixed .8",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    ROOT.mkdir(parents=True, exist_ok=True)
    with gzip.open(ROOT / f"{mode}.json.gz", "wt") as handle:
        json.dump(output, handle)
    print("COMPLETE", mode, output["metrics"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualify", action="store_true")
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    require_df2_backend("cpu")
    if args.qualify:
        qualify()
    else:
        run(args.index)
