"""Compare 1s/2oct, 1s/4oct and 1s/1000Hz across the seven frozen failures."""

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
from test_fading_diagonal import FadingDiagonal, fade_matrix
from test_octave_fading import OctaveFading, octave_surfaces
from test_octave_fading import signature as previous_signature

ROOT = Path("results/fading-failures")
DOCS = Path("docs/fading-failures")
VARIANTS = ("1000hz", "2oct", "4oct")
CASES = [dict(target_id="C04-T0005", events=4, target=6)] + json.loads(
    Path("docs/takeover-validation/cases.json").read_text()
)


def base_path(case):
    if case["target_id"] == "C04-T0005":
        return Path("docs/amplitude-pilot/baseline.json")
    return Path("results/takeover-validation/raw") / case["target_id"] / "baseline.json"


def signature():
    h = hashlib.sha256(previous_signature().encode() + Path(__file__).read_bytes())
    h.update(json.dumps(CASES, sort_keys=True).encode())
    for case in CASES:
        h.update(base_path(case).read_bytes())
    return h.hexdigest()


def frequency_matrix(hz, horizon):
    coordinate = torch.log2(hz.clamp_min(20) / 20)
    distance = (coordinate[:, None] - coordinate[None, :]).abs()
    return (1 - torch.log1p(9 * distance / horizon) / math.log(10)).clamp_min(0).tril()


class OctaveHorizon(OctaveFading):
    def __init__(self, target, horizon):
        self.base = CumulativeEnergyDistance(target)
        self.ordinary = False
        power = self.base._power(target[None])[0]
        self.mass = power.sum().detach()
        self.frequency = frequency_matrix(
            torch.arange(power.shape[0], dtype=torch.float64) * 4000 / 256, horizon
        )
        self.time = fade_matrix(power.shape[1], 64 / 4000, 1.0)
        self.reference = self.features(power).detach()


def objective(audio, variant):
    return (
        FadingDiagonal(audio, 1, 1000)
        if variant == "1000hz"
        else OctaveHorizon(audio, int(variant[0]))
    )


def qualify():
    torch.manual_seed(794)
    hz = torch.tensor([0, 15.625, 31.25, 62.5, 125, 500], dtype=torch.float64)
    p = torch.rand(6, 4, dtype=torch.float64, requires_grad=True)
    t = fade_matrix(4, 0.25, 1)
    for horizon in (2, 4):
        f = frequency_matrix(hz, horizon)
        expected = torch.zeros(4, 6, 4, dtype=torch.float64)
        for q, (tr, fr) in enumerate((tr, fr) for tr in (False, True) for fr in (False, True)):
            for k in range(6):
                for n in range(4):
                    for i in range(6):
                        for j in range(4):
                            if (i >= k if fr else i <= k) and (j >= n if tr else j <= n):
                                d = abs(math.log2(max(float(hz[k]), 20) / max(float(hz[i]), 20)))
                                wf = max(0, 1 - math.log1p(9 * d / horizon) / math.log(10))
                                wt = max(0, 1 - math.log1p(9 * abs(n - j) * 0.25) / math.log(10))
                                expected[q, k, n] += p[i, j] * wf * wt
        torch.testing.assert_close(octave_surfaces(p, f, t), expected, atol=1e-14, rtol=1e-14)
        assert torch.autograd.gradcheck(
            lambda x, kernel=f: octave_surfaces(x, kernel, t), (p,), fast_mode=True
        )
        # A dyadic frequency pair exactly horizon octaves apart loses all weight.
        assert float(f[4 if horizon == 2 else 5, 2]) < 1e-15
    synth = PhraseSynth()
    for case in CASES:
        meta, target = load_target(case["events"], case["target"])
        assert meta.target_id == case["target_id"] and base_path(case).exists()
    _, target = load_target(8, 1)
    audio = synth.render(target).detach()
    candidate = synth.render(initial_candidate(8, device="cpu")).detach().requires_grad_()
    direction = torch.randn_like(candidate)
    direction /= direction.norm()
    for variant in VARIANTS:
        obj = objective(audio, variant)
        same = audio.clone().requires_grad_()
        loss = obj(same)
        assert loss == 0 and torch.autograd.grad(loss, same)[0].abs().max() == 0
        analytical = (torch.autograd.grad(obj(candidate), candidate)[0] * direction).sum()
        numeric = (obj(candidate + 1e-5 * direction) - obj(candidate - 1e-5 * direction)) / 2e-5
        torch.testing.assert_close(analytical, numeric, rtol=1e-4, atol=1e-9)
    torch.testing.assert_close(
        OctaveHorizon(audio, 1)(candidate), OctaveFading(audio)(candidate), rtol=0, atol=0
    )
    DOCS.mkdir(parents=True, exist_ok=True)
    report = dict(
        passed=True,
        signature=signature(),
        cases=CASES,
        variants=VARIANTS,
        all_four_explicit_sums=True,
        octave_cutoffs=True,
        gradient_checks=True,
        previous_one_octave_exact=True,
        starts=["plateau", "fresh"],
    )
    (DOCS / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print("QUALIFIED", signature(), flush=True)


def run(index):
    assert 0 <= index < 42
    q = json.loads((DOCS / "qualification.json").read_text())
    assert q["passed"] and q["signature"] == signature()
    case = CASES[index // 6]
    mode = "plateau" if index % 6 < 3 else "fresh"
    variant = VARIANTS[index % 3]
    base = json.loads(base_path(case).read_text())
    initial = (
        initial_candidate(case["events"], device="cpu")
        if mode == "fresh"
        else EventPhrase(
            *(
                torch.tensor(base["best_phrase"][k], dtype=torch.float64)
                for k in ("f0_hz", "onset_seconds")
            )
        )
    )
    synth = PhraseSynth()
    meta, target = load_target(case["events"], case["target"])
    audio = synth.render(target).detach()
    obj = objective(audio, variant)

    def progress(s, _):
        if s.update % 500 == 0:
            print(index, case["target_id"], mode, variant, s.update, s.best_loss, flush=True)

    result = fit(
        audio,
        case["events"],
        "fading_failures",
        synth=synth,
        objective=obj,
        initial=initial,
        progress=progress,
    )
    with torch.no_grad():
        rendered = synth.render(result.best_phrase)
        assert abs(float(obj(rendered)) - result.best_loss) < 1e-10
        canonical = float(obj.base(rendered))
    out = dict(
        index=index,
        case=case,
        mode=mode,
        variant=variant,
        signature=signature(),
        fit=serialise_fit(result),
        metrics=describe(result.best_phrase, synth, audio, meta),
        initial_metrics=describe(initial, synth, audio, meta),
        canonical_loss=canonical,
        renderer=synth.provenance(),
        optimizer=asdict(PAPER_OPTIMIZER),
        time_horizon_seconds=1,
        frequency_horizon=1000 if variant == "1000hz" else int(variant[0]),
        frequency_unit="Hz" if variant == "1000hz" else "octaves",
        frequency_floor_hz=20,
        log_weighing=False,
        amplitudes="fixed .8",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / f"{index:02d}.json.gz"
    with gzip.open(path.with_suffix(".tmp.gz"), "wt") as handle:
        json.dump(out, handle)
    path.with_suffix(".tmp.gz").replace(path)
    print("COMPLETE", index, out["metrics"], flush=True)


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
