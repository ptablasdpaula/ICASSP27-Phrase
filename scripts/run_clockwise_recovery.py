"""Single-stage clockwise recovery, with the paper's Adam/patience schedule."""

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
from _clockwise_study_fit import fit
from icassp27_phrase.config import CARDINALITIES, PAPER_OPTIMIZER
from icassp27_phrase.runtime import require_df2_backend
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.targets import load_target
from run_direction_recovery import ROOT, Objective, describe, serialise_fit
from run_direction_recovery import signature as orthogonal_signature

ORDER = (0, 6, 1, 5, 3, 7, 2, 4)  # ↗ → ↘ ↓ ↙ ← ↖ ↑
PERIOD = 800


def signature():
    h = hashlib.sha256(orthogonal_signature().encode())
    for name in ("scripts/run_clockwise_recovery.py", "scripts/_clockwise_study_fit.py"):
        h.update(Path(name).read_bytes())
    return h.hexdigest()


def weights(update):
    position = (update % PERIOD) / (PERIOD / 8)
    sector = int(position)
    fraction = position - sector
    w = torch.zeros(8, dtype=torch.float64)
    w[ORDER[sector]] = 1 - fraction
    w[ORDER[(sector + 1) % 8]] = fraction
    return w


class ClockwiseObjective:
    def __init__(self, audio, weighted):
        self.base = Objective(audio, "diagonal", weighted)

    def __call__(self, audio, update):
        terms = self.base.all_terms(audio)
        return (terms * weights(update)).sum(), terms.mean()


def run(index):
    assert 1500 <= index < 3000
    weighted = index >= 2250
    variant = "clockwise_lw" if weighted else "clockwise"
    events = CARDINALITIES[(index % 750) // 150]
    metadata, target = load_target(events, index % 150 + 1)
    path = ROOT / "raw" / variant / f"{metadata.target_id}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    sig = signature()
    if path.exists():
        with gzip.open(path, "rt") as f:
            assert json.load(f)["signature"] == sig
        print("EXISTS", index, flush=True)
        return
    start = time.perf_counter()
    synth = PhraseSynth()
    with torch.no_grad():
        audio = synth.render(target)
    objective = ClockwiseObjective(audio, weighted)
    result = fit(audio, events, variant, synth=synth, objective=objective)
    final = result.best_phrase
    out = dict(
        index=index,
        variant=variant,
        schema="single-stage-clockwise-v1",
        target=asdict(metadata),
        signature=sig,
        renderer=synth.provenance(),
        optimizer=asdict(PAPER_OPTIMIZER),
        torch_version=torch.__version__,
        fit=serialise_fit(result),
        updates=result.updates,
        best_loss=result.best_loss,
        best_phrase=dict(f0_hz=final.f0_hz.tolist(), onset_seconds=final.onset_seconds.tolist()),
        metrics=describe(final, synth, audio, metadata),
        period=PERIOD,
        order=ORDER,
        diagonal_warmup=False,
        refinement=False,
        monitoring="fixed equal mean of all eight terms with matching Log-Weighing",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        target_registry_sha256=hashlib.sha256(
            Path("src/data/targets.json").read_bytes()
        ).hexdigest(),
        wall_seconds=time.perf_counter() - start,
    )
    with torch.no_grad():
        assert abs(float(objective(synth.render(final), 0)[1]) - result.best_loss) < 1e-10
    temporary = path.with_suffix(".tmp.gz")
    with gzip.open(temporary, "wt") as f:
        json.dump(out, f, separators=(",", ":"))
    temporary.replace(path)
    print(
        "COMPLETE",
        index,
        variant,
        metadata.target_id,
        result.updates,
        result.stopped_by,
        out["metrics"],
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=int, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    require_df2_backend("cpu")
    q = json.loads(Path("docs/direction-recovery-150/clockwise-qualification.json").read_text())
    assert q["passed"] and q["signature"] == signature()
    run(args.index)


if __name__ == "__main__":
    main()
