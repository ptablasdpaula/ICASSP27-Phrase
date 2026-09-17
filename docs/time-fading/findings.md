# Findings: time-only fading with entirely unfaded frequency

All14 fresh runs completed successfully. The time horizon changed from infinity
to0.25s; frequency accumulation stayed completely unfaded throughout every run.
No previously fitted phrase or saved plateau was used. All four diagonals,
fixed amplitudes and no Log-Weighing.

At the retained best common-unfaded-loss checkpoints, time-only fading improves
joint RMS error on5/7 phrases and worsens it on2/7. Median joint error falls from
.069321 to .051444, but mean rises from .106752 to .130069 because of the two
regressions. Thus the gain is not uniform and should not be described as an
overall mean improvement.

Two phrases meet both <1-cent pitch MAE and <1-ms onset MAE:

- C04-T0005: .187 cents / .036 ms, versus6.227/70.012 for the matched control.
- C04-T0003: .398 cents / .104 ms, versus43.191/101.795 for the matched control.

Both are also recovered at the actual last iterate (.161/.080 and .794/.287).
The matched unfaded control recovers neither. Compared with the preceding
both-axis fading schedule, this time-only version has2 recoveries instead of0,
and5/7 gains instead of3/7 under the same common-loss selection. The frequency
and time schedules differ, so this is not a complete factor-isolation ablation.

The significant regression is C04-T0007:12.133 cents /2.683 ms becomes
490.513/179.884. C08-T0004 also worsens:180.933/203.400 becomes203.423/299.703.
C06-T0000 has better joint error despite worse pitch MAE:7.308/45.614 becomes
46.093/36.291. The other gains include C06-T0008 (304.312/215.278 to49.864/180.308)
and C08-T0000 (86.430/43.842 to67.116/18.690).

The actual last iterates improve joint error on4/7 and worsen it on3/7.
Their mean and median are both worse than the matched control (.149418/.085276
versus .136297/.069111). Retained checkpoints therefore matter beyond the two
clear recoveries. Both selected and final-state tables are included.

Controls follow exactly the same optimiser timing:2000 updates with LR .05 and
no early stop, then patience up to3000. They are NOT the original immediate-
patience diagonal baseline. Their selected/final errors and selected parameters
reproduce the previous matched unfaded controls exactly (control-reproduction.json).

All source/signature/coverage checks passed. Frequency horizons are absent in
every recorded evaluation, and the qualification explicitly verifies unit causal
frequency weights. Initial/middle/final objective values and gradients match
the direct time-only reference; selected losses pass rerendering checks. This is
one prescribed schedule on seven selected failures, not representative validation.
