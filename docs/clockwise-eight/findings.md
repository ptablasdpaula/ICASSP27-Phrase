# Interpretation

The clockwise schedule is promising on this small, selected plateau set. Against
an equal-update diagonal control, it improves joint event error on 3/6 additional
failures, ties 2/6, and is worse on 1/6. All three gains exceed 20%; the regression
is 11.08%. Median relative joint-error reduction is 36.78%. Uniform eight-direction
averaging improves 2/6, ties 3/6 and worsens 1/6; its median reduction is zero.
These results exclude the development case from aggregation.

The largest rotation gains, pitch cents / onset milliseconds:

| Case | Diagonal control | Fixed eight | Clockwise |
|---|---:|---:|---:|
| C04-T0003 | 42.875 / 101.271 | 42.875 / 101.271 | 0.030 / 0.008 |
| C06-T0000 | 201.539 / 121.817 | 40.995 / 76.375 | 6.861 / 45.589 |
| C08-T0000 | 13.232 / 49.637 | 42.198 / 33.823 | 1.473 / 0.334 |

Only C04-T0003 crosses both strict recovery thresholds (<1 cent and <1 ms) among
the six additional cases. C08-T0000 comes close but does not meet the pitch threshold.
On the development case, rotation and fixed-eight averaging both recover the target,
whereas diagonal-only descent does not. Rotation is not needed for that particular
success in this run.

## When the useful states occurred

- C04-T0005 (development): selected update 2315, during diagonal refinement.
- C04-T0003: selected update 2253, during diagonal refinement.
- C06-T0000: selected update 2032, during diagonal refinement. The final rotated
  state had slightly worse canonical loss than the starting plateau, but polishing
  it yielded a substantially better result.
- C08-T0000: selected update **1127, during rotation**. Subsequent rotation lost
  that state. Refining the final rotated endpoint did not recover it: the final
  unselected iterate has 255.44 cents / 130.31 ms error. Retaining the best common
  diagonal-loss checkpoint is essential to this reported gain.
- The other three clockwise runs retain their initial plateau. On C08-T0004,
  the diagonal control improves joint error while rotation does not, so retaining
  the baseline still leaves rotation worse than the control.

A previous live progress message incorrectly attributed the eight-event gain to
endpoint refinement. The recorded best-update index establishes that it arose
during rotation; the results here and the final summary use that corrected account.

## Limits

This tests one clockwise period, one initial phase and a 100-update warm-up,
with fixed amplitudes, no Log-Weighing and no per-direction scale matching.
The warm-up is not separately ablated from rotation. All methods receive 1,600
exploration plus 800 refinement updates; equal updates do not imply equal runtime.

The [numerical sensitivity audit](numerical-sensitivity.json) is important: the
new fixed-eight development trajectory differs from the earlier eight-direction
pilot despite identical starting parameters and initial losses differing only at
round-off scale. Differences grow during optimisation. This does not isolate
summation order versus CPU execution effects, and one trajectory per case is not
a robustness test. Current paired variants share a process/node per case.

This is evidence that the schedule **can** help some stalled fits, not evidence of
a generally reliable training improvement or a reason to discard the incumbent.
No fresh-start 150-phrase training experiment has been run here.
