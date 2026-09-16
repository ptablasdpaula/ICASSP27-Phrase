"""Optimise all six pitch-swap proposals before choosing by canonical loss."""

import itertools
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
from _amplitude_fit import fit
from icassp27_phrase.config import EventPhrase
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.targets import load_target
from test_amplitude_recovery import metrics

ROOT = Path("docs/swap-escape-pilot/relaxed-swaps")


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    ROOT.mkdir(parents=True, exist_ok=True)
    prior = json.loads(Path("docs/amplitude-pilot/baseline.json").read_text())
    f0, onset = [
        torch.tensor(prior["best_phrase"][key], dtype=torch.float64)
        for key in ["f0_hz", "onset_seconds"]
    ]
    metadata, target = load_target(4, 6)
    synth = PhraseSynth()
    with torch.no_grad():
        audio = synth.render(target)
    # Fixed lexicographic enumeration; selection uses final loss, not target coordinates.
    for i, j in itertools.combinations(range(4), 2):
        path = ROOT / f"swap_{i + 1}{j + 1}.json"
        if path.exists():
            raise FileExistsError(path)
        candidate = f0.clone()
        candidate[i], candidate[j] = f0[j], f0[i]

        def progress(s, _audio, i=i, j=j):
            assert s.amplitudes == (0.8,) * 4
            if s.update % 400 == 0:
                print("swap", i + 1, j + 1, s.update, s.raw_loss, flush=True)

        r = fit(
            audio,
            4,
            "bidirectional_cumulative_energy",
            synth=synth,
            initial=EventPhrase(candidate, onset),
            free_amplitudes=False,
            progress=progress,
        )
        d = asdict(r)
        d["best_phrase"] = dict(
            f0_hz=r.best_phrase.f0_hz.tolist(), onset_seconds=r.best_phrase.onset_seconds.tolist()
        )
        d["metrics"] = metrics(
            r.best_phrase.f0_hz.numpy(), r.best_phrase.onset_seconds.numpy(), metadata
        )
        d["source_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        d["pair"] = [i + 1, j + 1]
        for s in d["trajectory"]:
            s["metrics"] = metrics(s["f0_hz"], s["onset_seconds"], metadata)
        path.write_text(json.dumps(d, indent=2) + "\n")
        print("COMPLETE", i + 1, j + 1, r.best_loss, d["metrics"], flush=True)


if __name__ == "__main__":
    main()
