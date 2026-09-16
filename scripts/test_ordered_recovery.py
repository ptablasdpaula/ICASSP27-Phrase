"""Four-event CeL recovery with cumulative positive onset gaps; fixed amplitude."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
from _amplitude_fit import decode_coordinates, encode_coordinates, fit
from icassp27_phrase.config import PAPER_OPTIMIZER, EventPhrase, initial_candidate
from icassp27_phrase.losses import build_loss
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.targets import load_target
from test_amplitude_recovery import metrics


def encode_ordered(f0, onset):
    """Represent N positive gaps plus terminal slack with N unconstrained logits."""
    gaps = torch.diff(torch.cat((onset.new_tensor([0.2]), onset, onset.new_tensor([1.8]))))
    if not bool((gaps > 0).all()):
        raise ValueError("Initial onsets must be strictly increasing inside (0.2, 1.8)")
    pitch = encode_coordinates(f0, onset)[:, 0]
    return torch.stack((pitch, torch.log(gaps[:-1] / gaps[-1])), dim=-1)


def decode_ordered(raw):
    """Onset i = 0.2 + sum of first i positive gaps; last onset stays below 1.8."""
    f0, _ = decode_coordinates(raw)
    fractions = torch.softmax(torch.cat((raw[:, 1], raw.new_zeros(1))), dim=0)
    onset = 0.2 + 1.6 * fractions[:-1].cumsum(0)
    return f0, onset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("docs/ordered-time-pilot"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    metadata, target = load_target(4, 6)
    synth = PhraseSynth()
    prior_path = Path("docs/amplitude-pilot/baseline.json")
    prior = json.loads(prior_path.read_text())
    stuck = EventPhrase(
        *(
            torch.tensor(prior["best_phrase"][key], dtype=torch.float64)
            for key in ["f0_hz", "onset_seconds"]
        )
    )
    with torch.no_grad():
        audio = synth.render(target)
    if args.verify:
        for phrase in [initial_candidate(4, device="cpu"), stuck, target]:
            raw = encode_ordered(phrase.f0_hz, phrase.onset_seconds).requires_grad_()
            f, t = decode_ordered(raw)
            torch.testing.assert_close(f, phrase.f0_hz, rtol=1e-14, atol=1e-14)
            torch.testing.assert_close(t, phrase.onset_seconds, rtol=1e-14, atol=1e-14)
            assert bool((torch.diff(t) > 0).all()) and 0.2 < t[0] < t[-1] < 1.8
            assert torch.autograd.gradcheck(decode_ordered, (raw,), eps=1e-6, atol=1e-7, rtol=1e-5)
            objective = build_loss("bidirectional_cumulative_energy", audio)
            loss = objective(synth.render_batch(f[None], t[None])[0])
            physical = torch.stack((f.detach(), t.detach()), -1).requires_grad_()
            direct = objective(synth.render_batch(physical[:, 0][None], physical[:, 1][None])[0])
            (gphysical,) = torch.autograd.grad(direct, physical)
            (gchain,) = torch.autograd.grad(
                torch.stack((f, t), -1), raw, grad_outputs=gphysical, retain_graph=True
            )
            (graw,) = torch.autograd.grad(loss, raw)
            torch.testing.assert_close(graw, gchain, rtol=1e-10, atol=1e-10)
        print(
            "PASS: ordered-coordinate round trips, bounds, "
            "Jacobian finite differences and loss chain rule"
        )
        return
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = dict(
        target=asdict(metadata),
        renderer=synth.provenance(),
        optimizer=asdict(PAPER_OPTIMIZER),
        device="cpu",
        torch=torch.__version__,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        parameterization=(
            "g = 1.6 softmax([z_1,...,z_N,0]); onset_i = 0.2 + sum(g[:i+1]); final g is end slack"
        ),
        amplitudes="fixed 0.8; never optimised",
        baseline_source=str(prior_path),
        loss="bidirectional_cumulative_energy: four directions, sqrt, no Log-Weighing",
    )
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for name, initial, expected_loss in [
        ("ordered_restart", stuck, prior["best_loss"]),
        ("ordered_initial", initial_candidate(4, device="cpu"), prior["initial_loss"]),
    ]:
        output = args.output / f"{name}.json"
        if output.exists():
            raise FileExistsError(output)

        def progress(s, _audio, name=name, expected_loss=expected_loss):
            if s.evaluation == 1:
                assert abs(s.raw_loss - expected_loss) < 1e-12, (s.raw_loss, expected_loss)
            assert s.amplitudes == (0.8,) * 4
            assert all(a < b for a, b in zip(s.onset_seconds, s.onset_seconds[1:], strict=False))
            if s.update % 100 == 0:
                print(name, s.update, s.raw_loss, flush=True)

        result = fit(
            audio,
            4,
            "bidirectional_cumulative_energy",
            synth=synth,
            initial=initial,
            free_amplitudes=False,
            progress=progress,
            coordinate_encoder=encode_ordered,
            coordinate_decoder=decode_ordered,
        )
        payload = asdict(result)
        payload["best_phrase"] = dict(
            f0_hz=result.best_phrase.f0_hz.tolist(),
            onset_seconds=result.best_phrase.onset_seconds.tolist(),
        )
        payload["metrics"] = metrics(
            result.best_phrase.f0_hz.numpy(), result.best_phrase.onset_seconds.numpy(), metadata
        )
        for s in payload["trajectory"]:
            s["metrics"] = metrics(s["f0_hz"], s["onset_seconds"], metadata)
        output.write_text(json.dumps(payload, indent=2) + "\n")
        print("COMPLETE", name, payload["metrics"], payload["best_phrase"], flush=True)


if __name__ == "__main__":
    main()
