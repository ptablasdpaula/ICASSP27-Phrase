# Findings across the seven failures

The preferred horizon depends on the starting state. With a fresh standard
initial guess, 1000 Hz gives the lowest mean and median joint RMS error:
mean/median .129882/.160345, versus .206250/.219674 for 2 octaves and
.162723/.176152 for 4 octaves. Relative to 1000 Hz, 2 octaves improves joint
error in 2/7 cases and worsens it in 5/7; 4 octaves improves 1/7 and worsens 6/7.
These are comparisons of joint RMS matched error, not simultaneous improvements
in both pitch and onset MAE.

At the saved plateaus, 2 octaves improves joint error versus 1000 Hz in 5/7 cases,
and 4 octaves in 4/7. Mean/median errors are .177300/.196357 for 2 octaves,
.189976/.196235 for 4 octaves, and .192565/.209734 for 1000 Hz. These gains
usually leave substantial residual errors; better than 1000 Hz does not imply
recovery or even an improvement over the original saved plateau.

Three individual fits meet both <1-cent pitch MAE and <1-ms onset MAE:

- C08-T0000, fresh, 1000 Hz: .084231 cents / .025082 ms.
- C08-T0000, plateau, 2 octaves: .057399 cents / .029298 ms.
- C04-T0003, fresh, 4 octaves: both errors below .001.

Thus the 2-octave horizon DOES escape the eight-event plateau where 1 octave
and 1000 Hz did not. But from a fresh start, the same 2-octave objective reaches
19.775 cents / 51.410 ms on that phrase. Four octaves solves C04-T0003 from a
fresh start while all its plateau runs remain inaccurate. There is no common
horizon that recovers the set; none of these objectives fully recovers either
six-event failure or C08-T0004 from either start.

After excluding the original C04-T0005 development case and the C08-T0000
fade-exploration case, each octave method beats 1000 Hz on joint error in 3/5
plateau cases and 1/5 fresh cases. Only the C04-T0003 fresh 4-octave fit meets
the recovery criterion on these remaining five. These are still selected
failures; this is not an independent representative benchmark.

All 42 Slurm tasks completed with exit0. All runs passed own-best rerendering,
source/coverage validation, finite metric checks and exact agreement of initial
metrics across the three objectives. The repeated C08-T0000 1000-Hz trajectories,
selected parameters and metrics match the previous fading pilot exactly for
both starts (reproduction.json). This validates the reused comparator here,
not cross-hardware robustness of every trajectory.

Only one trajectory per setting, fixed amplitudes, uniform final RMS (no
Log-Weighing), same patience and maximum budget. Actual update counts vary;
there is no forced continuation after patience. Full trajectories, per-case
metrics, mean/sampleSD/median and paired comparisons are saved alongside this note.
