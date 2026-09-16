"""Paired four-event amplitude pilot; run from the repository root."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from _amplitude_fit import fit, render_amplitudes
from icassp27_phrase.cel_screen import matching
from icassp27_phrase.config import PAPER_OPTIMIZER, initial_candidate
from icassp27_phrase.optimization import fit as registered_fit
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.targets import load_target


def metrics(f0, onset, target):
    f0, onset = np.asarray(f0), np.asarray(onset)
    tf, tt = np.asarray(target.f0_hz), np.asarray(target.onset_seconds)

    def coordinates(f, t):
        return np.stack((np.log2(f / 80) / 2, (t - 0.2) / 1.6), -1)

    assignment, tied = matching(coordinates(f0, onset), coordinates(tf, tt))
    return dict(
        pitch_mae_cents=float(np.abs(1200 * np.log2(f0 / tf[assignment])).mean()),
        onset_mae_ms=float(np.abs(1000 * (onset - tt[assignment])).mean()),
        assignment=assignment.tolist(),
        tied=tied,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/amplitude-pilot"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    synth = PhraseSynth()
    metadata, target = load_target(4, 6)
    with torch.no_grad():
        audio = synth.render(target)
    if args.verify:
        p = initial_candidate(4, device="cpu")
        f, t = (
            p.f0_hz.flip(0).clone().requires_grad_(),
            p.onset_seconds.flip(0).clone().requires_grad_(),
        )
        a = torch.full_like(f, 0.8, requires_grad=True)
        ordinary = synth.render_batch(f[None], t[None])[0]
        extended = render_amplitudes(synth, f, t, a)
        torch.testing.assert_close(ordinary, extended, rtol=0, atol=0)
        g0 = torch.autograd.grad(ordinary.square().sum(), (f, t))
        g1 = torch.autograd.grad(extended.square().sum(), (f, t, a))
        for x, y in zip(g0, g1[:2], strict=True):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        # Check amplitude differentiation numerically, including unsorted onsets.
        h = 1e-5
        for i in range(4):
            delta = torch.zeros_like(a)
            delta[i] = h
            with torch.no_grad():
                fd = (
                    render_amplitudes(synth, f, t, a + delta).square().sum()
                    - render_amplitudes(synth, f, t, a - delta).square().sum()
                ) / (2 * h)
            torch.testing.assert_close(fd, g1[2][i], rtol=1e-6, atol=1e-8)
        config = replace(PAPER_OPTIMIZER, maximum_updates=3)
        old = registered_fit(
            audio, 4, "bidirectional_cumulative_energy", synth=synth, config=config
        )
        new = fit(audio, 4, "bidirectional_cumulative_energy", synth=synth, config=config)
        assert old.best_loss == new.best_loss
        assert [s.raw_loss for s in old.trajectory] == [s.raw_loss for s in new.trajectory]
        print(
            "PASS: exact fixed-amplitude audio, control gradients and Adam trajectory; "
            "amplitude finite differences",
            flush=True,
        )
        return
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = dict(
        target=asdict(metadata),
        renderer=synth.provenance(),
        optimizer=asdict(PAPER_OPTIMIZER),
        torch=torch.__version__,
        device="cpu",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        amplitude_parameterization=(
            "a_i = 0.8 exp(z_i), z_i initially 0; same Adam LR; no regularizer"
        ),
        loss="bidirectional_cumulative_energy: four directions, sqrt feature, no Log-Weighing",
    )
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    baseline = None
    for name, free, restart in [
        ("baseline", False, False),
        ("restart_fixed", False, True),
        ("restart_amplitude", True, True),
        ("initial_amplitude", True, False),
    ]:
        path = args.output / f"{name}.json"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

        def progress(s, _audio, name=name):
            if s.update % 100 == 0:
                print(name, s.update, s.raw_loss, flush=True)

        result = fit(
            audio,
            4,
            "bidirectional_cumulative_energy",
            synth=synth,
            initial=baseline if restart else None,
            free_amplitudes=free,
            progress=progress,
        )
        if name == "baseline":
            baseline = result.best_phrase
        payload = asdict(result)
        payload["best_phrase"] = dict(
            f0_hz=result.best_phrase.f0_hz.tolist(),
            onset_seconds=result.best_phrase.onset_seconds.tolist(),
        )
        payload["metrics"] = metrics(
            payload["best_phrase"]["f0_hz"], payload["best_phrase"]["onset_seconds"], metadata
        )
        for snapshot in payload["trajectory"]:
            snapshot["metrics"] = metrics(snapshot["f0_hz"], snapshot["onset_seconds"], metadata)
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(
            "COMPLETE",
            name,
            payload["metrics"],
            payload["best_phrase"],
            result.best_amplitudes,
            flush=True,
        )


if __name__ == "__main__":
    main()
