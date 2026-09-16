"""Switch from the last orthogonal iterate back to diagonal-only CeL."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
from _amplitude_fit import fit
from _takeover_validation import Engine, initial_state
from icassp27_phrase.config import PAPER_OPTIMIZER, EventPhrase
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.targets import load_target
from validate_takeover_cases import save

ROOT = Path("results/last-orthogonal-refinement/raw")
SOURCE = Path("results/orthogonal-validation/raw")


def phrase(data):
    return EventPhrase(*(torch.tensor(data[k], dtype=torch.float64)
                         for k in ["f0_hz", "onset_seconds"]))


def run(index):
    case = json.loads(Path("docs/takeover-validation/cases.json").read_text())[index]
    root = ROOT / case["target_id"]
    root.mkdir(parents=True, exist_ok=True)
    old_summary = json.loads((SOURCE / case["target_id"] / "summary.json").read_text())
    save(root / "previous_summary.json", old_summary)
    metadata, target = load_target(case["events"], case["target"])
    synth = PhraseSynth()
    with torch.no_grad():
        audio = synth.render(target)
    engine = Engine(synth, audio, metadata, old_summary["initial"]["loss"])
    perm = torch.arange(case["events"])
    for variant in ["all_four", "frequency_only"]:
        output = root / f"{variant}.json"
        if output.exists():
            continue
        source = SOURCE / case["target_id"] / f"{variant}.json"
        old = json.loads(source.read_text())
        last = old["exploration_trajectory"][-1]
        assert last["update"] == 1600
        initial = phrase(last)
        before = engine.describe(initial_state(initial), perm)
        assert abs(before["loss"] - last["canonical_loss"]) < 1e-12
        print("START", case["target_id"], variant, before["loss"], before["metrics"], flush=True)

        def progress(snapshot, _audio, variant=variant):
            assert snapshot.amplitudes == (.8,) * case["events"]
            if snapshot.update % 500 == 0:
                print(case["target_id"], variant, snapshot.update, snapshot.best_loss, flush=True)

        fitted = fit(audio, case["events"], "bidirectional_cumulative_energy", synth=synth,
                     initial=initial, free_amplitudes=False, progress=progress)
        result = asdict(fitted)
        result["best_phrase"] = dict(f0_hz=fitted.best_phrase.f0_hz.tolist(),
                                     onset_seconds=fitted.best_phrase.onset_seconds.tolist())
        final = engine.describe(initial_state(fitted.best_phrase), perm)
        assert abs(final["loss"] - fitted.best_loss) < 1e-12
        assert fitted.best_loss <= before["loss"] + 1e-12
        assert abs(fitted.initial_loss - before["loss"]) < 1e-12
        payload = dict(case=case, variant=variant, source_path=str(source),
                       source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                       source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                              text=True).strip(),
                       target=asdict(metadata), renderer=synth.provenance(),
                       optimizer=asdict(PAPER_OPTIMIZER), torch=torch.__version__,
                       device="cpu", start_update=1600, start=before, final=final, fit=result,
                       previous=old_summary["variants"][variant],
                       original_stall=old_summary["initial"],
                       previous_control=old_summary["control"]["final"],
                       previous_control_budget=old_summary["control_budget"])
        save(output, payload)
        print("COMPLETE", case["target_id"], variant, final["loss"], final["metrics"],
              "updates", fitted.updates, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", nargs="+", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    for index in args.indices:
        run(index)


if __name__ == "__main__":
    main()
