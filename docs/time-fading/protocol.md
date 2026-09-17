# Time-only fading, frequency entirely unfaded

User correction: NO frequency fade. Shrink only the time fade from infinity
to0.25 seconds. The initial interpretation of retaining1000Hz fading was
corrected before any fits were launched. No result from that interpretation
is included here.

Seven previously tested problem cases, with standard fresh initialisation:
C04-T0005, C04-T0003, C04-T0007, C06-T0000, C06-T0008, C08-T0000, C08-T0004.
One time-annealed fit and one matched fully unfaded control per phrase;14 fits.
Slurm array27215846. Fixed amplitudes .8; no Log-Weighing.

## Exact accumulation and schedule

All four diagonals. Frequency accumulation is the ordinary inclusive cumulative
sum throughout: every causal frequency-bin contribution has unit weight. The
implementation uses a triangular all-ones matrix (and the reverse orientation).
There is no frequency decay, finite horizon, octave distance or local frequency
normalisation. The STFT stays on its original linear-Hz grid.

Time contributions alone use the prior logarithmic kernel
w(d;H)=max(0,1-log(1+9d/H)/log(10)), with d and H in seconds. H is the zero-weight
horizon; half-weight is about .2403H. Normalise by original global target power,
apply the usual square-root feature/floor1e-12 and average four directional RMSs.

Update0 is exactly the original unfaded diagonal objective. During updates1–100,
increase inverse time horizon linearly from zero to1/748.036139 seconds. This
first finite anchor retains99% of energy at the full1.936-second STFT separation.
Then decrease H geometrically from748.036139 seconds at update100 to0.25 seconds
at update2000. Frequency weights remain identically one at every update.

The current controls and Adam moments persist continuously through this schedule.
No previous fit, saved plateau, pre-fit or additional restart initialises training.
From update2000, keep time horizon0.25s and enable standard relative improvement
1e-4, patience100/200 rollback/clear moments/LR x.3, stop250 or total3000 updates.
Both variants use LR .05 before that, and scale by the identical initial loss.

## Matched controls and selection

The controls have NO fade along either axis. They use precisely the same fresh
initial guess and optimisation timing:2000 updates before enabling patience,
maximum3000. Thus these controls differ from the earlier150-target runs that
used patience from the start. Actual post2000 update counts may differ.

Both methods report their strict-best common ORIGINAL UNFADED diagonal loss
checkpoint across the entire run, without parameter-error selection. Selection
may retain a pre-terminal-horizon state. Also report the actual last iterate
separately. During final-phase rollback, each method uses its own fixed training
objective. The stored trajectory fields canonical_loss and best_canonical_loss
refer to the original unfaded diagonal loss, not a frequency-faded comparator.

## Verification and reporting

Qualification verifies identically unit causal frequency weights, monotonically
shrinking time horizons, the99% initial finite-anchor calibration, and exact
value/gradient agreement with explicit time-only fading at initial, middle and
final horizons. Both self-loss/self-gradient and audio finite differences pass.
The previously qualified optimiser engine is reused unchanged, including its
terminal patience and rollback tests. Best selected losses are rerendered.

Report all seven per-case matched pitch/onset MAEs, joint RMS error, LSD, update
counts, best and last states. Aggregates also exclude C04-T0005/C08-T0000, the
previous development/exploration examples. One trajectory per setting on selected
failures, with no broader generalisation or robustness claim.
