"""Validate single-stage rotation and exact reuse of registered patience/Adam rules."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch
from _clockwise_study_fit import fit
from icassp27_phrase.config import PAPER_OPTIMIZER
from icassp27_phrase.losses import build_loss
from icassp27_phrase.optimization import fit as registered_fit
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.targets import load_target
from run_clockwise_recovery import ORDER, ClockwiseObjective, signature, weights


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    synth = PhraseSynth()
    _, target = load_target(2, 1)
    audio = synth.render(target).detach()
    name = "bidirectional_cumulative_energy"
    fixed = build_loss(name, audio)

    def stationary(candidate, update):
        loss = fixed(candidate)
        return loss, loss

    for config in (
        replace(PAPER_OPTIMIZER, maximum_updates=12),
        replace(
            PAPER_OPTIMIZER,
            maximum_updates=25,
            plateau_patience=2,
            stop_patience=5,
            meaningful_relative_improvement=0.999999,
        ),
    ):
        old = registered_fit(audio, 2, name, synth=synth, config=config)
        new = fit(audio, 2, name, synth=synth, config=config, objective=stationary)
        assert old.best_loss == new.best_loss and old.updates == new.updates
        for a, b in zip(old.trajectory, new.trajectory, strict=True):
            assert (a.raw_loss, a.learning_rate, a.patience, a.plateau_events) == (
                b.raw_loss,
                b.learning_rate,
                b.patience,
                b.plateau_events,
            )
    for step in range(3001):
        w = weights(step)
        assert (w >= 0).all() and abs(float(w.sum()) - 1) < 1e-14
        torch.testing.assert_close(w, weights(step + 800), rtol=0, atol=0)
    for i, direction in enumerate(ORDER):
        expected = torch.zeros(8, dtype=torch.float64)
        expected[direction] = 1
        torch.testing.assert_close(weights(100 * i), expected, rtol=0, atol=0)
    for weighted in (False, True):
        objective = ClockwiseObjective(audio, weighted)
        candidate = torch.roll(audio, 3).requires_grad_()
        terms = objective.base.all_terms(candidate)
        for step in (0, 50, 100, 750):
            train, monitor = objective(candidate, step)
            torch.testing.assert_close(monitor, terms.mean(), rtol=0, atol=0)
            selected = weights(step).nonzero().flatten()
            explicit = sum(terms[i] * weights(step)[i] for i in selected)
            g = torch.autograd.grad(train, candidate, retain_graph=True)[0]
            expected = torch.autograd.grad(explicit, candidate, retain_graph=True)[0]
            torch.testing.assert_close(g, expected, rtol=1e-10, atol=1e-12)
            assert torch.isfinite(g).all()
        fitted = fit(
            audio,
            2,
            "clockwise",
            synth=synth,
            objective=objective,
            config=replace(PAPER_OPTIMIZER, maximum_updates=5),
        )
        assert fitted.updates == 5
        assert fitted.best_loss == min(s.raw_loss for s in fitted.trajectory)
    report = dict(
        passed=True,
        signature=signature(),
        stationary_optimizer_exact=True,
        rollback_patience_exact=True,
        period=800,
        no_warmup=True,
        no_refinement=True,
        training_only_gradient_checked=True,
        weighted_and_unweighted_checked=True,
        monitor="equal mean of eight terms with matching weighting",
    )
    Path("docs/direction-recovery-150/clockwise-qualification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
