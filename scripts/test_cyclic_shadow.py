"""Cyclic one-step pitch-assignment probes; only real-branch gradients are committed."""

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
from _amplitude_fit import decode_coordinates, encode_coordinates, fit
from icassp27_phrase.config import EventPhrase
from icassp27_phrase.losses import build_loss
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.targets import load_target
from test_amplitude_recovery import metrics
from test_shadow_swaps import State, advance, score

ROOT = Path("docs/shadow-swap-pilot")
STEPS = 300


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    baseline = json.loads(Path("docs/amplitude-pilot/baseline.json").read_text())
    f, t = [
        torch.tensor(baseline["best_phrase"][key], dtype=torch.float64)
        for key in ["f0_hz", "onset_seconds"]
    ]
    metadata, target = load_target(4, 6)
    synth = PhraseSynth()
    with torch.no_grad():
        audio = synth.render(target)
    objective = build_loss("bidirectional_cumulative_energy", audio)
    scale = baseline["best_loss"]
    for name, enabled in [("cyclic_one_step", True), ("no_probe_control", False)]:
        path = ROOT / f"{name}.json"
        if path.exists():
            raise FileExistsError(path)
        raw = encode_coordinates(f, t)
        real = State(raw, torch.zeros_like(raw), torch.zeros_like(raw))
        permutation = torch.arange(4)
        best = scale
        best_state = real.clone()
        best_perm = permutation.clone()
        records = []
        for step in range(STEPS):
            before = real.clone()
            updated, _ = advance(real, permutation, objective, synth, scale)
            keep_loss = score(updated, permutation, objective, synth)
            row = dict(update=step + 1, keep_loss=keep_loss, accepted=False)
            if enabled:
                i = step % 3
                j = i + 1
                proposal = permutation.clone()
                proposal[i], proposal[j] = permutation[j], permutation[i]
                trial, _ = advance(before, proposal, objective, synth, scale)
                trial_loss = score(trial, proposal, objective, synth)
                probabilities = torch.softmax(
                    torch.tensor([-keep_loss / scale, -trial_loss / scale], dtype=torch.float64), 0
                )
                row.update(
                    pair=[i + 1, j + 1],
                    trial_loss=trial_loss,
                    swap_probability=float(probabilities[1]),
                )
                # Exactly one pitch per event. Softmax is over two whole permutations.
                if trial_loss < keep_loss:
                    permutation = proposal
                    row["accepted"] = True
                # Backward of trial must not modify either live state or its real update.
                for attr in ["raw", "first", "second"]:
                    assert torch.equal(getattr(real, attr), getattr(before, attr))
                    assert getattr(real, attr).grad is None
            real = updated
            actual = score(real, permutation, objective, synth)
            assert sorted(permutation.tolist()) == list(range(4))
            if actual < best:
                best = actual
                best_state = real.clone()
                best_perm = permutation.clone()
            pf, pt = decode_coordinates(real.raw)
            row.update(
                actual_loss=actual,
                best_loss=best,
                permutation=permutation.tolist(),
                metrics=metrics(pf[permutation].numpy(), pt.numpy(), metadata),
            )
            records.append(row)
            if step % 50 == 0:
                print(name, step + 1, best, sum(r["accepted"] for r in records), flush=True)
        pf, pt = decode_coordinates(best_state.raw)
        polish = fit(
            audio,
            4,
            "bidirectional_cumulative_energy",
            synth=synth,
            initial=EventPhrase(pf[best_perm], pt),
            free_amplitudes=False,
        )
        out = asdict(polish)
        out["best_phrase"] = dict(
            f0_hz=polish.best_phrase.f0_hz.tolist(),
            onset_seconds=polish.best_phrase.onset_seconds.tolist(),
        )
        out["metrics"] = metrics(
            polish.best_phrase.f0_hz.numpy(), polish.best_phrase.onset_seconds.numpy(), metadata
        )
        payload = dict(
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            exploratory_updates=STEPS,
            records=records,
            best_loss=best,
            polish=out,
            selection="hard argmax of two-assignment softmax, temperature=baseline loss",
            persistence="assignment only; trial parameter and moment updates discarded",
        )
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print("COMPLETE", name, best, out["metrics"], flush=True)


if __name__ == "__main__":
    main()
