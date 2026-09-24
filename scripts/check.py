"""Small numerical and installation checks; no pytest or full campaign required."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from icassp27_phrase import PhraseSynth
from icassp27_phrase.data import load_target
from icassp27_phrase.data.design import candidates, target_unit_design, targets
from icassp27_phrase.losses import NAMES, PaperObjectives
from icassp27_phrase.metrics import (
    hungarian_assignment,
    log_spectral_distance,
    phrase_gradient_cosine,
)
from icassp27_phrase.optimization import SCHEDULE, fit_batch, retain_strict_best
from icassp27_phrase.paths import resolve_path, validate_reference
from icassp27_phrase.runtime import configure_reproducibility, require_df2_backend
from icassp27_phrase.synth.config import CARDINALITIES, WaveguideConfig
from icassp27_phrase.synth.controls import encode_defaults, render_all_controls


def sampling_checks():
    for count in CARDINALITIES:
        design = target_unit_design(count)
        strata = np.floor(design.reshape(32, -1) * 32).astype(int)
        np.testing.assert_array_equal(
            np.sort(strata, axis=0), np.broadcast_to(np.arange(32)[:, None], strata.shape)
        )
        assert len([load_target(count, i + 1) for i in range(150)]) == 150
    for name, target in targets():
        assert len(target) == 1 or np.diff(target[:, 1]).min() >= 0.05 - 1e-12
        designs = candidates(name, target)
        np.testing.assert_array_equal(designs["joint"], candidates(name, target)["joint"])
        np.testing.assert_array_equal(
            designs["pitch"][..., 1], np.broadcast_to(target[:, 1], (256, len(target)))
        )
        np.testing.assert_array_equal(
            designs["time"][..., 0], np.broadcast_to(target[:, 0], (256, len(target)))
        )
    permutation, tied = hungarian_assignment(
        np.array([[1.05, 1], [0, 0.05]]), np.array([[0, 0], [1, 1]])
    )
    np.testing.assert_array_equal(permutation, [1, 0])
    assert not tied
    target = np.array([[0.0, 0.0]])
    candidate = np.array([[[0.3, 0.4]]])
    assignment = np.array([[0]])
    for sign in (1, -1):
        cosine = phrase_gradient_cosine(
            sign * candidate[:, None], candidate, target, assignment, np.array([False])
        )
        np.testing.assert_allclose(cosine, [[sign]])
    assert SCHEDULE.threshold == 0.0001 and SCHEDULE.maximum_updates == 20000
    best = torch.tensor([3.0, 2.0])
    raw = torch.zeros(2, 1, 2)
    retain_strict_best(
        best, raw, torch.tensor([0, 1]), torch.tensor([1.0, 4.0]), torch.ones_like(raw)
    )
    assert best.tolist() == [1.0, 2.0] and raw[0].sum() == 2 and raw[1].sum() == 0
    print("Sampling, matching, cosine, lowest-loss retention: OK", flush=True)


def scheduling_checks():
    class ConstantObjective:
        def __init__(self, target, name):
            pass

        def __call__(self, audio, indices):
            return 1 + audio.sum(-1) * 0

    schedule = replace(SCHEDULE, plateau_patience=1, stop_patience=5, maximum_updates=10)
    with patch("icassp27_phrase.optimization.PairedObjective", ConstantObjective):
        result = fit_batch(
            torch.ones(2, 1, dtype=torch.float64),
            1,
            "cel",
            lambda f, t: f + t * 0,
            schedule=schedule,
        )
    assert result[5].tolist() == [5, 5]
    assert result[6].tolist() == [2, 2]
    assert [event["update"] for event in result[7][0]] == [2, 4]
    assert [event["new_lr"] for event in result[7][0]] == [0.025, 0.0125]
    print("Plateau timing and independent early stopping: OK", flush=True)


def progress_checks():
    class QuadraticObjective:
        def __init__(self, target, name):
            self.target = target

        def __call__(self, audio, indices):
            return 1 + (audio - self.target[indices]).square().sum(-1)

    renders, snapshots = [], []

    def synth(f, t):
        audio = f + t
        renders.append(audio.detach().clone())
        return audio

    def progress(snapshot, audio):
        assert not audio.requires_grad
        torch.testing.assert_close(audio, renders[-1][0], rtol=0, atol=0)
        assert snapshot.evaluation == snapshot.update + 1
        snapshots.append(snapshot)

    with patch("icassp27_phrase.optimization.PairedObjective", QuadraticObjective):
        for cap, patience, expected in ((23, 1000, 23), (23, 3, 3)):
            schedule = replace(
                SCHEDULE,
                maximum_updates=cap,
                stop_patience=patience,
                threshold=0.99 if patience == 3 else SCHEDULE.threshold,
            )
            snapshots.clear()
            renders.clear()
            args = (torch.ones(1, 1, dtype=torch.float64), 1, "cel", synth)
            reported = fit_batch(*args, schedule=schedule, progress=progress)
            assert [s.update for s in snapshots] == list(range(expected + 1))
            assert len(renders) == expected + 1  # No extra synthesis for the display.
            silent = fit_batch(*args, schedule=schedule)
            for actual, reference in zip(reported[:7], silent[:7], strict=True):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    print("Per-iteration progress, terminal callbacks and unchanged optimisation: OK", flush=True)


def numerical_checks(device: str, baseline: Path | None):
    configure_reproducibility()
    torch.set_num_threads(1)
    require_df2_backend(device)
    output = {}
    for count in (1, 2):
        synth = PhraseSynth().to(device)
        f = torch.tensor(
            [[147.3, 231.7][:count]], dtype=torch.float64, device=device, requires_grad=True
        )
        t = torch.tensor(
            [[0.4031, 1.1023][:count]], dtype=torch.float64, device=device, requires_grad=True
        )
        with torch.no_grad():
            target = synth(f * 1.03, t + 0.02)
        audio = synth(f, t)
        values = PaperObjectives(target, NAMES).values(audio)
        output[f"audio{count}"] = audio.detach().cpu().numpy()
        for i, (name, value) in enumerate(values.items()):
            gradient = torch.autograd.grad(value.sum(), (f, t), retain_graph=i < len(values) - 1)
            assert bool(torch.isfinite(value).all()) and all(
                bool(torch.isfinite(g).all()) for g in gradient
            )
            output[f"{count}_{name}"] = np.concatenate(
                [value.detach().cpu().numpy().ravel(), *(g.cpu().numpy().ravel() for g in gradient)]
            )
        assert float(log_spectral_distance(audio.detach(), audio.detach())) == 0
        if count == 1:
            fit = fit_batch(target, 1, "cel", synth, schedule=replace(SCHEDULE, maximum_updates=3))
            output["fit"] = np.concatenate([x.cpu().numpy().ravel() for x in fit[:7]])
            assert bool((fit[3] <= fit[4]).all()) and int(fit[5][0]) == 3
        print(f"{count}-event synthesis, all losses and gradients: OK", flush=True)
    persistent = PhraseSynth(
        waveguide_config=replace(WaveguideConfig(), state_policy="persistent")
    ).to(device)
    output["persistent"] = persistent(f, t).detach().cpu().numpy()
    unit = encode_defaults(device=device).requires_grad_(True)
    audio = render_all_controls(unit)
    output["controls_audio"] = audio.detach().cpu().numpy()
    output["controls_grad"] = torch.autograd.grad(audio.square().mean(), unit)[0].cpu().numpy()
    assert np.isfinite(output["controls_grad"]).all()
    if baseline:
        with np.load(baseline, allow_pickle=False) as saved:
            if set(saved.files) != set(output):
                raise ValueError("Baseline keys differ")
            for key, value in output.items():
                np.testing.assert_allclose(value, saved[key], rtol=1e-9, atol=1e-10, err_msg=key)
        print(
            "Pre-refactor baseline: within 1e-9 relative / 1e-10 absolute tolerance",
            flush=True,
        )
    print("Persistent state and seven-control synthesis: OK", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument(
        "--data-only",
        action="store_true",
        help="Validate saved data and sampling without synthesis",
    )
    p.add_argument("--baseline", type=resolve_path, help="Optional pre-refactor numerical NPZ")
    a = p.parse_args()
    validate_reference()
    sampling_checks()
    scheduling_checks()
    progress_checks()
    if not a.data_only:
        numerical_checks(a.device, a.baseline)
    print("All requested checks passed", flush=True)


if __name__ == "__main__":
    main()
