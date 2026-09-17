# Full orthogonal / clockwise phrase-recovery study

**Running:** 3,000 fits, comprising 150 frozen targets at each of 1/2/4/6/8 events,
for orthogonal, orthogonal + Log-Weighing, clockwise and clockwise + Log-Weighing.

All four methods start from the standard initial guess, with the same Adam,
patience and 3,000-update cap. Clockwise rotates throughout: no diagonal pre-fit,
warm-up or refinement. Its fixed eight-direction mean monitors progress and
selects checkpoints; only the rotating loss supplies gradients.

- Orthogonal array: `27187528` (resumes valid completed fits).
- Corrected clockwise gate: `27188071`; full array: `27188083`.
- Reporting/publishing job: `27188097`, dependent on both arrays succeeding.
- Maximum 64 concurrent full-study CPU fits.
- Optimiser, rollback and gradient checks passed for both clockwise weightings.

See the [protocol](protocol.md), [original qualification](qualification.json),
[clockwise qualification](clockwise-qualification.json), and [job metadata](jobs.json).
Selected states, per-phrase metrics, aggregate mean/SD/median tables and plots
will be published here automatically after all 3,000 fits validate. Compressed
trajectories stay under `results/direction-recovery-150/raw`. The cancelled staged
clockwise outputs are archived separately and excluded.

These are not completed results yet. The paper is unchanged.
