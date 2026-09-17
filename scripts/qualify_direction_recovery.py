"""Fail closed before launching the full recovery study."""

from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from _recovery_study_fit import fit
from icassp27_phrase.config import PAPER_OPTIMIZER
from icassp27_phrase.losses import build_loss
from icassp27_phrase.optimization import fit as registered_fit
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.targets import load_target
from run_direction_recovery import Objective, log_spectral_distance, signature


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    synth = PhraseSynth()
    _, target = load_target(2, 1)
    audio = synth.render(target).detach()
    config = replace(PAPER_OPTIMIZER, maximum_updates=12)
    name = "bidirectional_cumulative_energy"
    old = registered_fit(audio, 2, name, synth=synth, config=config)
    new = fit(audio, 2, name, synth=synth, config=config, objective=build_loss(name, audio))
    assert old.best_loss == new.best_loss
    assert [s.raw_loss for s in old.trajectory] == [s.raw_loss for s in new.trajectory]
    # Exercise rollback, LR reduction and patience, not just the initial Adam steps.
    config = replace(
        config,
        maximum_updates=25,
        plateau_patience=2,
        stop_patience=5,
        meaningful_relative_improvement=0.999999,
    )
    old = registered_fit(audio, 2, name, synth=synth, config=config)
    new = fit(audio, 2, name, synth=synth, config=config, objective=build_loss(name, audio))
    assert old.stopped_by == new.stopped_by == "patience"
    assert [s.raw_loss for s in old.trajectory] == [s.raw_loss for s in new.trajectory]
    for weighted in (False, True):
        for family in ("orthogonal", "diagonal"):
            objective = Objective(audio, family, weighted)
            same = audio.clone().requires_grad_()
            value = objective(same)
            assert value == 0 and torch.autograd.grad(value, same)[0].abs().max() == 0
            candidate = (audio * 0.9).requires_grad_()
            terms = objective.all_terms(candidate)
            expected = terms[4:].mean() if family == "orthogonal" else terms[:4].mean()
            torch.testing.assert_close(objective(candidate), expected, rtol=0, atol=0)
            assert torch.isfinite(torch.autograd.grad(expected, candidate)[0]).all()
    # Execute the original paper's metric function verbatim, without importing its application.
    path = Path(
        "/data/home/acw794/ICASSP2027-PluckedStringPhrase-linear-jtfot-confirmatory"
        "/src/plucked_string_phrase/pilot.py"
    )
    tree = ast.parse(path.read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "log_spectral_distance"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    candidate = torch.roll(audio, 3) * 0.8
    expected = float(namespace["log_spectral_distance"](candidate[None], audio[None])[0])
    assert log_spectral_distance(candidate, audio) == expected
    assert log_spectral_distance(audio, audio) == 0
    assert all(len(load_target(n, i)[0].f0_hz) == n for n in (1, 2, 4, 6, 8) for i in range(1, 151))
    report = dict(
        passed=True,
        signature=signature(),
        torch_version=torch.__version__,
        numpy_version=np.__version__,
        optimizer_exact_agreement=True,
        plateau_schedule_exact_agreement=True,
        lsd_exact_agreement=True,
        objectives_checked=["orthogonal", "orthogonal_lw", "diagonal", "diagonal_lw"],
        targets=750,
    )
    Path("docs/direction-recovery-150/qualification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
