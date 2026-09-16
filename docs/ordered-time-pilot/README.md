# Cumulative onset-gap pilot, fixed excitation amplitude

Follow-up to the [amplitude pilot](../amplitude-pilot/README.md), requested
2026-09-16: test chronological onsets formed by cumulative positive increments
on the same failed four-event target C04-T0005 (UI target 6). **All excitation
amplitudes remain fixed at 0.8; amplitudes are not learned.**

## Parameterisation

For four free gap logits z, form five positive gaps:

`g = 1.6 * softmax([z1, z2, z3, z4, 0])`

The onsets are `0.2 + g1`, `0.2 + g1 + g2`, and so on. The fifth gap is
unused terminal slack, keeping the fourth onset below 1.8 s. Thus event 1
occurs at t1 and event i occurs at t1 plus the subsequent positive increments.
This is a bounded version of the requested cumulative-time parameterisation,
with four free timing parameters, strict chronological order, no clipping and
no minimum inter-event spacing. Gap normalisation couples the timing controls.
Pitch uses the same independent bounded log-frequency logits as before.

The two new fits start (1) at the previous fixed-amplitude baseline's strict-best
swapped controls and (2) at the registered equal-cell initialisation. Both use
four-direction unweighted square-root CeL (implementation name
`bidirectional_cumulative_energy`), FFT length 16384 for 8000 samples (2.048),
float64 CPU synthesis, and the registered Adam schedule: initial LR 0.05,
initial-loss conditioning, rollback/reduced LR on plateaus and at most 3000
updates. Each run starts with fresh moments. No LR tuning is performed for
the new coordinate system.

The [previous fixed restart](../amplitude-pilot/restart_fixed.json) and
[original baseline](../amplitude-pilot/baseline.json) are the independent-time
controls; their stored results are reused, not rerun. New fits check that the
initial physical loss matches the respective control to absolute 1e-12 and
that onsets stay ordered and amplitudes stay 0.8 at every evaluation. Coordinates
are encoded to reproduce the same initial pitches and times.

Validation checks coordinate round trips, the coordinate Jacobian by finite
differences, and the rendered-loss chain rule. The prior amplitude-pilot
verification also checks that the shared experimental fitter still reproduces
the production fixed-amplitude audio, gradients and initial Adam trajectory.
Production experiment code and paper text are unchanged.

Run with the project Python and PYTHONHOME set to `.pixi/envs/default`:

```
python scripts/test_ordered_recovery.py --verify
python scripts/test_ordered_recovery.py
```

Results pending. This selected-case diagnostic cannot establish performance
across the full 150-target recovery experiment.
