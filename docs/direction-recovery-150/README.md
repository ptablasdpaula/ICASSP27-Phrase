# Full orthogonal / clockwise / fixed-fading recovery study

**Running:** six methods × 750 targets = 4,500 fits. Each method uses 150 frozen
targets at each of 1/2/4/6/8 events.

- Orthogonals, without and with Log-Weighing (1500 fits completed).
- Clockwise rotation, without and with Log-Weighing (running).
- Fixed logarithmic fade 1 s / 1000 Hz over four diagonals, without and with Log-Weighing (added).

The new fading variants use the original registered Adam/patience settings from
the standard initial guess. No fade schedule, delayed patience, plateau restart
or refinement. The logarithmic fade kernel is shared by both; Log-Weighing changes
only the final quadrature. Each fixed objective selects its own best loss iterate.

[Protocol](protocol.md) · [Original qualification](qualification.json) ·
[Clockwise qualification](clockwise-qualification.json) ·
[Fading qualification](fading-qualification.json) · [Jobs](jobs.json)

Fading gate 27192164 precedes array 27192172. Combined report job 27192224 waits for
the clockwise and fading arrays, then validates all 4,500 fits including completed
orthogonals. It publishes mean/SD/median summaries, per-phrase results, comparisons
and plots, and automatically commits and pushes this directory. The previous
four-method report job was replaced; no experiment jobs were cancelled.

Full trajectories remain under `results/direction-recovery-150/raw`.
These are not completed aggregate results. The paper is unchanged.
