"""Qualify uniform and Log-Weighed fixed fading before the full campaign."""

import json
import math
from dataclasses import replace

import torch
from _recovery_study_fit import fit
from icassp27_phrase.config import PAPER_OPTIMIZER, initial_candidate
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.targets import load_target
from run_direction_recovery import signature as shared_signature
from run_fading_recovery import DOCS, FixedFading, signature
from test_fading_diagonal import FadingDiagonal


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(794)
    old = json.loads((DOCS / "qualification.json").read_text())
    assert old["passed"] and old["signature"] == shared_signature()
    synth = PhraseSynth()
    _, target = load_target(2, 1)
    audio = synth.render(target).detach()
    candidate = synth.render(initial_candidate(2, device="cpu")).detach().requires_grad_()
    original = FadingDiagonal(audio, 1.0, 1000.0)
    uniform = FixedFading(audio, False)
    torch.testing.assert_close(uniform(candidate), original(candidate), rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(uniform(candidate), candidate)[0],
        torch.autograd.grad(original(candidate), candidate)[0],
        rtol=0,
        atol=0,
    )
    config = replace(PAPER_OPTIMIZER, maximum_updates=12)
    expected = fit(audio, 2, "old", synth=synth, objective=original, config=config)
    actual = fit(audio, 2, "new", synth=synth, objective=uniform, config=config)
    assert expected.trajectory == actual.trajectory
    for weighted in (False, True):
        obj = FixedFading(audio, weighted)
        same = audio.clone().requires_grad_()
        loss = obj(same)
        assert loss == 0 and torch.autograd.grad(loss, same)[0].abs().max() == 0
        if weighted:
            # Independent quadrature construction on the physical seconds/octaves grid.
            error = obj.features(obj.base._power(candidate[None])[0]) - obj.reference
            nf, nt = error.shape[-2:]
            f = torch.log2((torch.arange(nf, dtype=torch.float64) * 4000 / 256).clamp_min(20) / 20)
            t = torch.arange(nt, dtype=torch.float64) * 0.016
            terms = []
            for q, (tr, fr) in enumerate((tr, fr) for tr in (False, True) for fr in (False, True)):
                area = torch.zeros(nf, nt, dtype=torch.float64)
                for k in range(nf):
                    k0, k1 = (k - 1, k) if fr else (k, k + 1)
                    if k0 < 0 or k1 >= nf:
                        continue
                    for n in range(nt):
                        n0, n1 = (n - 1, n) if tr else (n, n + 1)
                        if n0 < 0 or n1 >= nt:
                            continue
                        area[k, n] = (f[k1] - f[k0]) * (t[n1] - t[n0])
                terms.append((error[q].square() * area).sum().sqrt() / area.sum().sqrt())
            torch.testing.assert_close(
                obj(candidate), torch.stack(terms).mean(), rtol=1e-13, atol=1e-14
            )
        direction = torch.randn_like(candidate)
        direction /= direction.norm()
        analytic = (torch.autograd.grad(obj(candidate), candidate)[0] * direction).sum()
        numeric = (obj(candidate + 1e-5 * direction) - obj(candidate - 1e-5 * direction)) / 2e-5
        torch.testing.assert_close(analytic, numeric, rtol=1e-4, atol=1e-9)
        result = fit(audio, 2, "fading", synth=synth, objective=obj, config=config)
        assert all(math.isfinite(s.raw_loss) for s in result.trajectory)
    assert all(len(load_target(n, i)[0].f0_hz) == n for n in (1, 2, 4, 6, 8) for i in range(1, 151))
    report = dict(
        passed=True,
        signature=signature(),
        uniform_pilot_values_gradients_trajectory_exact=True,
        independent_log_quadrature=True,
        finite_difference_both=True,
        self_loss_gradient_both=True,
        registered_optimizer_and_lsd_qualification_current=True,
        targets=750,
        variants=2,
        time_horizon_seconds=1,
        frequency_horizon_hz=1000,
        fade_schedule=False,
    )
    (DOCS / "fading-qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    main()
