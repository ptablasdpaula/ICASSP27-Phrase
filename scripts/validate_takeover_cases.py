"""Fresh baseline, full takeover, and gradient-budget-matched ordinary control."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch
from _takeover_validation import (
    Engine,
    descend,
    initial_state,
    matched_control,
    paired_sweep,
    serialize_state,
)
from icassp27_phrase.config import PAPER_OPTIMIZER, EventPhrase
from icassp27_phrase.optimization import fit
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.targets import load_target


def signature():
    paths = sorted(Path("src").glob("*.py")) + [
        Path("src/data/targets.json"),
        Path("scripts/_amplitude_fit.py"),
        Path("scripts/_takeover_validation.py"),
        Path("scripts/validate_takeover_cases.py"),
        Path("scripts/test_counterfactual_takeover.py"),
        Path("scripts/test_amplitude_recovery.py"),
    ]
    digest = hashlib.sha256()
    for p in paths:
        digest.update(str(p).encode())
        digest.update(p.read_bytes())
    return digest.hexdigest()


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int)
    parser.add_argument("--qualify", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/takeover-validation/raw"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    synth = PhraseSynth()
    if args.qualify:
        old = json.loads(Path("docs/counterfactual-takeover/result.json").read_text())
        baseline = json.loads(Path("docs/amplitude-pilot/baseline.json").read_text())
        metadata, target = load_target(4, 6)
        with torch.no_grad():
            audio = synth.render(target)
        phrase = EventPhrase(
            *(
                torch.tensor(baseline["best_phrase"][k], dtype=torch.float64)
                for k in ["f0_hz", "onset_seconds"]
            )
        )
        engine = Engine(synth, audio, metadata, baseline["best_loss"])
        state = initial_state(phrase)
        _, _, records = paired_sweep(
            engine, state, torch.arange(4), progress=lambda s: print(s, flush=True)
        )
        for actual, expected in zip(records, old["trials"], strict=True):
            assert actual["depth"] == expected["depth"]
            assert actual["chosen"] == expected["chosen"]
            assert actual["accepted_swap"] == expected["accepted_swap"]
            for k in ["best_keep", "best_swap"]:
                assert abs(actual[k] - expected[k]) < 1e-12
            assert abs(actual["after"]["loss"] - expected["live_after"]["loss"]) < 1e-12
        # Verify matched control actually spends its allocated budget and preserves the best.
        control_engine = Engine(synth, audio, metadata, baseline["best_loss"])
        best, _ = matched_control(
            control_engine, state, torch.arange(4), 7, progress=lambda s: None
        )
        assert control_engine.backwards == 7
        assert control_engine.value(best, torch.arange(4)) <= baseline["best_loss"] + 1e-12
        args.output.mkdir(parents=True, exist_ok=True)
        save(
            args.output / "qualification.json",
            dict(
                source_signature=signature(),
                passed=True,
                checks=(
                    "original four-event paired decisions/depths/losses reproduced to 1e-12; "
                    "exact trial isolation and takeover assertions; "
                    "matched control budget and checkpoint preservation"
                ),
            ),
        )
        print("QUALIFICATION PASSED", flush=True)
        return
    if args.index is None:
        parser.error("--index is required unless --qualify")
    qualification = json.loads((args.output / "qualification.json").read_text())
    sig = signature()
    assert qualification["passed"] and qualification["source_signature"] == sig
    case = json.loads(Path("docs/takeover-validation/cases.json").read_text())[args.index]
    output = args.output / case["target_id"]
    output.mkdir(parents=True, exist_ok=True)
    metadata, target = load_target(case["events"], case["target"])
    provenance = dict(
        case=case,
        target=asdict(metadata),
        renderer=synth.provenance(),
        optimizer=asdict(PAPER_OPTIMIZER),
        source_signature=sig,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        device="cpu",
        torch=torch.__version__,
        amplitudes="fixed 0.8",
        baseline="fresh registered equal-cell initialisation, 2.048 padding",
        control="same additional backward-update budget, restart ordinary Adam on patience",
    )
    if (output / "provenance.json").exists():
        assert json.loads((output / "provenance.json").read_text())["source_signature"] == sig
    else:
        save(output / "provenance.json", provenance)
    with torch.no_grad():
        audio = synth.render(target)
    baseline_path = output / "baseline.json"
    if baseline_path.exists():
        base = json.loads(baseline_path.read_text())
    else:
        print("BASELINE", case["target_id"], flush=True)

        def progress(s, _audio):
            if s.update % 500 == 0:
                print("baseline", s.update, s.best_loss, flush=True)

        started = time.perf_counter()
        result = fit(
            audio, case["events"], "bidirectional_cumulative_energy", synth=synth, progress=progress
        )
        base = asdict(result)
        base["best_phrase"] = dict(
            f0_hz=result.best_phrase.f0_hz.tolist(),
            onset_seconds=result.best_phrase.onset_seconds.tolist(),
        )
        base["measured_wall_seconds"] = time.perf_counter() - started
        save(baseline_path, base)
    phrase = EventPhrase(
        *(
            torch.tensor(base["best_phrase"][k], dtype=torch.float64)
            for k in ["f0_hz", "onset_seconds"]
        )
    )
    start = initial_state(phrase)
    permutation = torch.arange(case["events"])
    engine = Engine(synth, audio, metadata, base["best_loss"])
    initial = engine.describe(start, permutation)
    assert abs(initial["loss"] - base["best_loss"]) < 1e-12
    save(output / "initial.json", initial)
    print("FRESH BASELINE", case["target_id"], initial["metrics"], flush=True)
    heuristic_path = output / "heuristic.json"
    if heuristic_path.exists():
        heuristic = json.loads(heuristic_path.read_text())
    else:
        started = time.perf_counter()
        state, perm, trials = paired_sweep(
            engine, start, permutation, progress=lambda s: print(s, flush=True)
        )
        adopted = engine.describe(state, perm)
        state, continuation = descend(engine, state, perm, progress=lambda s: print(s, flush=True))
        final = engine.describe(state, perm)
        heuristic = dict(
            trials=trials,
            adopted=adopted,
            final=final,
            continuation=continuation,
            state=serialize_state(state, perm),
            backward_updates=engine.backwards,
            forward_renders=engine.forwards,
            wall_seconds=time.perf_counter() - started,
        )
        save(heuristic_path, heuristic)
    control_path = output / "control.json"
    if control_path.exists():
        control = json.loads(control_path.read_text())
    else:
        print("MATCHED CONTROL", heuristic["backward_updates"], "updates", flush=True)
        control_engine = Engine(synth, audio, metadata, base["best_loss"])
        started = time.perf_counter()
        best, attempts = matched_control(
            control_engine,
            start,
            permutation,
            heuristic["backward_updates"],
            progress=lambda s: print("control", s, flush=True),
        )
        final = control_engine.describe(best, permutation)
        control = dict(
            final=final,
            attempts=attempts,
            state=serialize_state(best, permutation),
            backward_updates=control_engine.backwards,
            forward_renders=control_engine.forwards,
            wall_seconds=time.perf_counter() - started,
        )
        save(control_path, control)
    assert control["backward_updates"] == heuristic["backward_updates"]
    summary = dict(
        case=case,
        initial=initial,
        heuristic=heuristic["final"],
        control=control["final"],
        accepted_swaps=sum(t["accepted_swap"] for t in heuristic["trials"]),
        gradient_budget=heuristic["backward_updates"],
        heuristic_wall_seconds=heuristic["wall_seconds"],
        control_wall_seconds=control["wall_seconds"],
        source_signature=sig,
    )
    save(output / "summary.json", summary)
    print("COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
