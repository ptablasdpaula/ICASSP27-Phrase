"""Full 150-per-cardinality recovery with fixed 1s/1000Hz logarithmic fading."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch
from _recovery_study_fit import fit
from icassp27_phrase.config import CARDINALITIES, PAPER_OPTIMIZER
from icassp27_phrase.runtime import require_df2_backend
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.targets import load_target
from run_direction_recovery import ROOT, describe, serialise_fit
from run_direction_recovery import signature as shared_signature
from test_fading_diagonal import FadingDiagonal

DOCS = Path("docs/direction-recovery-150")


def signature():
    h = hashlib.sha256(shared_signature().encode())
    for p in (Path(__file__), Path("scripts/test_fading_diagonal.py")):
        h.update(p.read_bytes())
    return h.hexdigest()


class FixedFading(FadingDiagonal):
    def __init__(self, target, weighted=False):
        super().__init__(target, 1.0, 1000.0)
        self.weighted = weighted

    def __call__(self, audio):
        if not self.weighted:
            return super().__call__(audio)
        error = self.features(self.base._power(audio[None])[0]) - self.reference
        return torch.linalg.vector_norm((error * self.base.sqrt_weights).flatten(1), dim=1).mean()


def run(index):
    assert 3000 <= index < 4500
    sig = signature()
    q = json.loads((DOCS / "fading-qualification.json").read_text())
    assert q["passed"] and q["signature"] == sig
    weighted = index >= 3750
    variant = "fading_lw" if weighted else "fading"
    events = CARDINALITIES[(index % 750) // 150]
    meta, target = load_target(events, index % 150 + 1)
    path = ROOT / "raw" / variant / f"{meta.target_id}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with gzip.open(path, "rt") as f:
            previous = json.load(f)
        assert previous["signature"] == sig and previous["index"] == index
        print("EXISTS", index, flush=True)
        return
    started = time.perf_counter()
    synth = PhraseSynth()
    audio = synth.render(target).detach()
    objective = FixedFading(audio, weighted)
    result = fit(audio, events, variant, synth=synth, objective=objective)
    best = result.best_phrase
    with torch.no_grad():
        assert abs(float(objective(synth.render(best))) - result.best_loss) < 1e-10
    out = dict(
        index=index,
        variant=variant,
        schema="fixed-fading-v1",
        target=asdict(meta),
        signature=sig,
        fit=serialise_fit(result),
        updates=result.updates,
        best_loss=result.best_loss,
        best_phrase=dict(f0_hz=best.f0_hz.tolist(), onset_seconds=best.onset_seconds.tolist()),
        metrics=describe(best, synth, audio, meta),
        optimizer=asdict(PAPER_OPTIMIZER),
        renderer=synth.provenance(),
        time_horizon_seconds=1.0,
        frequency_horizon_hz=1000.0,
        fade_kernel="max(0, 1-log1p(9*d/H)/log(10)); separable time/frequency product",
        log_weighing=weighted,
        fade_schedule=False,
        amplitudes="fixed .8",
        selection="strict best own fixed training objective",
        wall_seconds=time.perf_counter() - started,
        torch_version=torch.__version__,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    temporary = path.with_suffix(".tmp.gz")
    with gzip.open(temporary, "wt") as f:
        json.dump(out, f, separators=(",", ":"))
    temporary.replace(path)
    print(
        "COMPLETE",
        index,
        variant,
        meta.target_id,
        result.updates,
        result.stopped_by,
        out["metrics"],
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=int, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    require_df2_backend("cpu")
    run(args.index)
