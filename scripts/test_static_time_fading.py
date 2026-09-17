"""Fixed .25-second time-only fading versus unfaded diagonals, seven fresh failures."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
from _recovery_study_fit import fit
from icassp27_phrase.config import PAPER_OPTIMIZER, EventPhrase
from icassp27_phrase.losses import CumulativeEnergyDistance
from icassp27_phrase.runtime import require_df2_backend
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.targets import load_target
from run_direction_recovery import describe, serialise_fit
from run_direction_recovery import signature as shared_signature
from test_fading_diagonal import FadingDiagonal
from test_time_fading import CASES
from test_time_fading import signature as previous_signature

ROOT = Path("results/static-time-fading")
DOCS = Path("docs/static-time-fading")


def signature():
    return hashlib.sha256(previous_signature().encode() + Path(__file__).read_bytes()).hexdigest()


def qualify():
    prior = json.loads(Path("docs/time-fading/qualification.json").read_text())
    assert prior["passed"] and prior["signature"] == previous_signature()
    registered = json.loads(Path("docs/direction-recovery-150/qualification.json").read_text())
    assert registered["passed"] and registered["signature"] == shared_signature()
    synth = PhraseSynth()
    _, target = load_target(4, 6)
    audio = synth.render(target).detach()
    obj = FadingDiagonal(audio, 0.25, None)
    assert torch.equal(obj.frequency, torch.ones_like(obj.frequency).tril())
    same = audio.clone().requires_grad_()
    value = obj(same)
    assert value == 0 and torch.autograd.grad(value, same)[0].abs().max() == 0
    DOCS.mkdir(parents=True, exist_ok=True)
    d = dict(
        passed=True,
        signature=signature(),
        previous_time_only_value_gradient_checks_current=True,
        registered_optimizer_checks_current=True,
        frequency_unit_weights=True,
        time_horizon_seconds=0.25,
        frequency_horizon=None,
        fade_schedule=False,
    )
    (DOCS / "qualification.json").write_text(json.dumps(d, indent=2) + "\n")
    print(d, flush=True)


def run(index):
    assert 0 <= index < 14
    q = json.loads((DOCS / "qualification.json").read_text())
    assert q["passed"] and q["signature"] == signature()
    case = CASES[index // 2]
    variant = "static_time" if index % 2 else "unfaded"
    synth = PhraseSynth()
    meta, target = load_target(case["events"], case["target"])
    audio = synth.render(target).detach()
    canonical = CumulativeEnergyDistance(audio)
    objective = FadingDiagonal(audio, 0.25, None) if index % 2 else canonical
    common_best = {}

    def progress(snapshot, candidate):
        with torch.no_grad():
            value = float(canonical(candidate))
        if not common_best or value < common_best["loss"]:
            common_best.update(
                loss=value,
                update=snapshot.update,
                f0_hz=list(snapshot.f0_hz),
                onset_seconds=list(snapshot.onset_seconds),
            )
        if snapshot.update % 500 == 0:
            print(index, variant, snapshot.update, snapshot.best_loss, flush=True)

    result = fit(
        audio, case["events"], variant, synth=synth, objective=objective, progress=progress
    )
    best = result.best_phrase
    common = EventPhrase(
        *(torch.tensor(common_best[k], dtype=torch.float64) for k in ("f0_hz", "onset_seconds"))
    )
    last = EventPhrase(
        *(
            torch.tensor(getattr(result.trajectory[-1], k), dtype=torch.float64)
            for k in ("f0_hz", "onset_seconds")
        )
    )
    with torch.no_grad():
        assert abs(float(objective(synth.render(best))) - result.best_loss) < 1e-10
        assert abs(float(canonical(synth.render(common))) - common_best["loss"]) < 1e-10
        canonical_loss = float(canonical(synth.render(best)))
    d = dict(
        index=index,
        case=case,
        variant=variant,
        signature=signature(),
        fit=serialise_fit(result),
        metrics=describe(best, synth, audio, meta),
        common_metrics=describe(common, synth, audio, meta),
        last_metrics=describe(last, synth, audio, meta),
        common_best=common_best,
        canonical_loss=canonical_loss,
        optimizer=asdict(PAPER_OPTIMIZER),
        renderer=synth.provenance(),
        time_horizon_seconds=0.25 if index % 2 else None,
        frequency_horizon_hz=None,
        fade_schedule=False,
        log_weighing=False,
        amplitudes="fixed .8",
        selection="strict best own fixed training loss; common-unfaded checkpoint also recorded",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / f"{index:02d}.json.gz"
    temporary = path.with_suffix(".tmp.gz")
    with gzip.open(temporary, "wt") as f:
        json.dump(d, f, separators=(",", ":"))
    temporary.replace(path)
    print("COMPLETE", index, case["target_id"], variant, d["metrics"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qualify", action="store_true")
    p.add_argument("--index", type=int)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    require_df2_backend("cpu")
    qualify() if args.qualify else run(args.index)
