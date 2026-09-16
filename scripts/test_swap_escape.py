"""Selected-case direction changes and objective-selected pitch-swap escape."""

from __future__ import annotations

import itertools
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
from _amplitude_fit import fit
from icassp27_phrase.config import PAPER_OPTIMIZER, EventPhrase
from icassp27_phrase.losses import build_loss
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.targets import load_target
from test_amplitude_recovery import metrics

ROOT = Path("docs/swap-escape-pilot")
VARIANTS = ["cel_01", "cel_04", "cel_05", "cel_10", "cel_05_lw", "cel_15_lw", "cel_03_lw"]


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    ROOT.mkdir(parents=True, exist_ok=True)
    metadata, target = load_target(4, 6)
    baseline = json.loads(Path("docs/amplitude-pilot/baseline.json").read_text())
    stuck = EventPhrase(
        *(
            torch.tensor(baseline["best_phrase"][key], dtype=torch.float64)
            for key in ["f0_hz", "onset_seconds"]
        )
    )
    synth = PhraseSynth()
    with torch.no_grad():
        audio = synth.render(target)
    canonical = build_loss("bidirectional_cumulative_energy", audio)
    proposals = []
    phrases = [stuck]
    with torch.no_grad():
        proposals.append(dict(pair=None, loss=float(canonical(synth.render(stuck)))))
        for i, j in itertools.combinations(range(4), 2):
            f = stuck.f0_hz.clone()
            f[i], f[j] = stuck.f0_hz[j], stuck.f0_hz[i]
            p = EventPhrase(f, stuck.onset_seconds)
            phrases.append(p)
            proposals.append(dict(pair=[i + 1, j + 1], loss=float(canonical(synth.render(p)))))
    chosen = min(range(len(proposals)), key=lambda i: proposals[i]["loss"])
    (ROOT / "swap_proposals.json").write_text(
        json.dumps(
            dict(
                proposals=proposals,
                chosen=chosen,
                selection="minimum canonical CeL, including unchanged state",
            ),
            indent=2,
        )
        + "\n"
    )
    provenance = dict(
        target=asdict(metadata),
        renderer=synth.provenance(),
        optimizer=asdict(PAPER_OPTIMIZER),
        device="cpu",
        torch=torch.__version__,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        amplitudes="fixed 0.8",
        times="independent bounded logits",
        variants=VARIANTS,
        start="docs/amplitude-pilot/baseline.json strict-best controls",
    )
    (ROOT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    cases = [("pitch_swap", "bidirectional_cumulative_energy", phrases[chosen])]
    cases += [(v, v, stuck) for v in VARIANTS]
    for name, loss_name, initial in cases:
        path = ROOT / f"{name}.json"
        if path.exists():
            raise FileExistsError(path)

        def progress(s, _audio, name=name):
            assert s.amplitudes == (0.8,) * 4
            if s.update % 200 == 0:
                print(name, s.update, s.raw_loss, flush=True)

        result = fit(
            audio,
            4,
            loss_name,
            synth=synth,
            initial=initial,
            free_amplitudes=False,
            progress=progress,
        )
        payload = asdict(result)
        payload["best_phrase"] = dict(
            f0_hz=result.best_phrase.f0_hz.tolist(),
            onset_seconds=result.best_phrase.onset_seconds.tolist(),
        )
        payload["metrics"] = metrics(
            result.best_phrase.f0_hz.numpy(), result.best_phrase.onset_seconds.numpy(), metadata
        )
        with torch.no_grad():
            payload["canonical_cel_at_best"] = float(canonical(synth.render(result.best_phrase)))
        for s in payload["trajectory"]:
            s["metrics"] = metrics(s["f0_hz"], s["onset_seconds"], metadata)
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print("COMPLETE", name, payload["metrics"], payload["canonical_cel_at_best"], flush=True)


if __name__ == "__main__":
    main()
