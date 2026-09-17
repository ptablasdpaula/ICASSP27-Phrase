# Findings: gradual fading versus matched fresh unfaded controls

This particular continuous schedule is not an overall improvement. Under the
predeclared common-unfaded-loss checkpoint selection, fading lowers joint RMS
error on3/7 cases and raises it on4/7. Mean joint error rises from .106752 to
.163689; median rises from .069321 to .114180. Neither method meets the combined
<1-cent/<1-ms recovery criterion on any of the seven cases.

The strongest gains are C06-T0008 (304.312 cents /215.278 ms to15.128/80.641)
and C08-T0004 (180.933/203.400 to124.995/52.562). C04-T0005 improves only slightly.
Substantial regressions include C04-T0007 (12.133/2.683 to496.668/171.832) and
C06-T0000 (7.308/45.614 to50.712/153.652). C08-T0000 is worse, unlike the earlier
successful fixed1000Hz fresh run or fixed2oct plateau restart.

Actual last iterates produce4/7 gains and3/7 regressions in joint error, but
mean still rises from .136297 to .157262 and median from .069111 to .136943.
No last iterate meets the recovery criterion either. For example, the faded
C04-T0003 endpoint has lower parameter errors than its common-loss checkpoint
(11.418/82.892 versus46.439/102.553), illustrating that choosing by the original
loss can disagree with parameter recovery. Both views are therefore reported.

Excluding C04-T0005 and C08-T0000, common-loss checkpoints improve2/5 and worsen3/5;
last iterates improve3/5 and worsen2/5. Mean joint errors still worsen in both
views. These five are selected failures, not a representative validation set.

Interpretation: continuously localising the objective can help individual
phrases, but this fixed transition from effectively global to1s/1oct does not
consistently steer training to the correct events. This does not rule out every
patience-triggered schedule or another transition speed, and it does not test
those alternatives. There was one schedule, not a hyperparameter search.

The unfaded controls use the SAME2000-update period without patience, followed
by terminal patience up to3000, as the fading runs. Thus these control results
must not be confused with earlier immediate-patience diagonal fits. In particular,
additional optimisation changes some original failure outcomes substantially.
No saved plateau or fitted parameters were used to initialise any run.

All14 Slurm tasks completed with exit0. Paired initial coordinates and loss
matched exactly; common-best checkpoint losses were checked by rerendering;
all annealed runs reached1s/1oct. Qualification includes fade calibration,
initial/terminal objective identities, audio gradients, and terminal patience
and rollback. Full trajectories and final states are retained. Existing paper
and production loss files are unchanged.
