"""Run two axis-only switches and a control with no smaller gradient budget."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from _takeover_validation import Engine, initial_state, matched_control, serialize_state
from icassp27_phrase.config import EventPhrase
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.targets import load_target
from validate_takeover_cases import save

ROOT = Path("results/orthogonal-validation/raw")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    case = json.loads(Path("docs/takeover-validation/cases.json").read_text())[args.index]
    root = ROOT / case["target_id"]
    root.mkdir(parents=True, exist_ok=True)
    old_root = Path("results/takeover-validation/raw") / case["target_id"]
    base = json.loads((old_root / "baseline.json").read_text())
    assert base["stopped_by"] == "patience" and base["trajectory"][-1]["patience"] >= 250
    synth = PhraseSynth()
    metadata, target = load_target(case["events"], case["target"])
    with torch.no_grad():
        audio = synth.render(target)
    phrase = EventPhrase(*(torch.tensor(base["best_phrase"][k], dtype=torch.float64)
                           for k in ["f0_hz", "onset_seconds"]))
    initial = initial_state(phrase)
    perm = torch.arange(case["events"])
    engine = Engine(synth, audio, metadata, base["best_loss"])
    initial_description = engine.describe(initial, perm)
    assert abs(initial_description["loss"] - base["best_loss"]) < 1e-12
    save(root / "baseline.json", base)
    save(root / "initial.json", initial_description)
    save(root / "baseline_provenance.json", json.loads((old_root / "provenance.json").read_text()))
    variants = {}
    budgets = {}
    for variant in ["all_four", "frequency_only"]:
        path = root / f"{variant}.json"
        if not path.exists():
            subprocess.run([sys.executable, "scripts/test_orthogonal_directions.py", variant,
                            "--case-index", str(args.index)], check=True)
        data = json.loads(path.read_text())
        budgets[variant] = 1600 + data["polish"]["updates"]
        final_phrase = EventPhrase(*(torch.tensor(data["polish"]["best_phrase"][k],
                                                  dtype=torch.float64)
                                    for k in ["f0_hz", "onset_seconds"]))
        final = engine.describe(initial_state(final_phrase), perm)
        assert abs(final["loss"] - data["polish"]["best_loss"]) < 1e-12
        variants[variant] = dict(final=final, updates=budgets[variant],
                                 best_exploration_loss=data["best_exploration_loss"])
    budget = max(budgets.values())
    control_path = root / "control.json"
    if control_path.exists():
        control = json.loads(control_path.read_text())
    else:
        control_engine = Engine(synth, audio, metadata, base["best_loss"])
        started = time.perf_counter()
        best, attempts = matched_control(control_engine, initial, perm, budget,
                                          progress=lambda s: print("control", s, flush=True))
        control = dict(final=control_engine.describe(best, perm), attempts=attempts,
                       updates=control_engine.backwards, forward_renders=control_engine.forwards,
                       wall_seconds=time.perf_counter() - started,
                       state=serialize_state(best, perm))
        save(control_path, control)
    assert control["updates"] == budget
    summary = dict(case=case, initial=initial_description, variants=variants, control=control,
                   control_budget=budget)
    save(root / "summary.json", summary)
    print("COMPLETE", case["target_id"], json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
