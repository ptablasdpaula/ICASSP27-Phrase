# Full orthogonal / clockwise phrase-recovery study

**Running:** 3,000 fits, comprising 150 frozen targets at each of 1/2/4/6/8 events,
for orthogonal, orthogonal + Log-Weighing, clockwise and clockwise + Log-Weighing.

- The fitter and original LSD metric passed exact-agreement qualification.
- All eight end-to-end checks (all four variants at one and eight events) passed.
- Main Slurm array: `27186997`, maximum 64 concurrent CPU fits.
- Reporting/publishing job: `27187049`, dependent on successful completion.

See the [protocol](protocol.md), [qualification](qualification.json), and
[job metadata](jobs.json). Clockwise is the plateau-triggered scheme: its extra
updates are reported explicitly. Selected states, per-phrase metrics, aggregate
mean/SD/median tables and plots will be published here automatically after all
3,000 fits validate. Full compressed trajectories stay under
`results/direction-recovery-150/raw`.

These are not completed results yet. The paper is unchanged.
