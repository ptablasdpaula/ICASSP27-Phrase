# Fresh scheduled fading experiment

User requested increasing fading from no effect towards 1 second / 1 octave,
and explicitly allowed a schedule other than patience. Final choice: continuous
update-based annealing. Patience on a changing loss risks interpreting changes
in the objective itself as optimisation progress. No fitted phrase or old
plateau initialises any run. All seven cases use the standard initial guess.
Slurm array 27191831: seven annealed fits and seven matched unfaded controls.

## Calibration and schedule

Use the previous logarithmic contribution kernel:

    w(d; H) = max(0, 1 - log(1+9*d/H)/log(10)).

Time/frequency weights multiply. Frequency separation is in octaves, using
log2(max(f,20)/20); final RMS remains uniform (no Log-Weighing). At the full
STFT spans (1.936 s and 6.643856 octaves), horizons 10 s / 40 octaves retain only
.338780 of a contribution after both scans. They are not approximately unfaded.

Update0: exact original four-diagonal cumulative loss (infinite horizons).
Updates1–100: increase inverse horizons linearly from zero to calibrated finite
horizons: 1504.734127 seconds and 5163.862162 octaves. Each full-axis contribution
then retains .995; both scans retain .990025. This is the explicit <1% attenuation
criterion for the first finite anchor, not a claim of exact equivalence.
Updates100–2000: decrease both horizons geometrically to 1 s / 1 octave. Equivalent
anchor pairs at updates100/480/860/1240/1620/2000 are:

- 1504.734127 s / 5163.862162 oct
- 348.311503 s / 934.070622 oct
- 80.626139 s / 168.960344 oct
- 18.663105 s / 30.562569 oct
- 4.320082 s / 5.528342 oct
- 1 s / 1 oct

These are points on a continuous curve, not independent fits or restarts.
Maintain current parameters and Adam state throughout the fade transition.
Updates2000 onwards: keep the 1 s / 1 octave loss fixed, enabling the usual
relative-improvement threshold1e-4, patience100/200 rollback/clear moments/LR x.3,
stop250 or the overall update3000 cap. Thus every fit reaches the terminal fade.
No gradient conditions, target parameters, or parameter-error metrics choose
when to fade or stop.

## Matched control and checkpoint reporting

The unfaded controls use precisely the same fresh starts, Adam and timing:
LR .05 with no rollback or early stop for the first2000 updates, then the same
patience logic on the fixed unfaded loss for at most1000 further updates. Both
methods condition gradients by the identical initial unfaded loss. These controls
differ from the earlier registered schedule which enabled patience immediately.
Actual update counts may differ after2000; both have the same maximum budget.

Each run retains the strict-best COMMON UNFADED diagonal loss across its entire
trajectory. This checkpoint is for final reporting and does not cause rollback
during annealing. Final-phase rollback uses that phase's own fixed objective.
Also report the actual last iterate independently, to expose whether an earlier
checkpoint masks deterioration. Best checkpoints may occur before1s/1oct.
No parameter-error selection. Retaining a checkpoint does not initialise a
second continuation run.

Fixed .8 amplitudes, independent bounded pitch/onset, 2.048 onset padding,
STFT256/hop64/periodicHann/center=False, square-root features, original global
mass, CPU float64, deterministic algorithms, one thread. All four diagonals.

## Verification and limitations

Exact unfaded update0, >=99% first-anchor retention, continuous horizon boundaries,
exact terminal agreement with the earlier 1s/1oct loss, audio gradient finite
differences and zero self-loss/gradient are checked. A synthetic constant-loss
run verifies reaching the final horizon and terminal patience/rollback; a short
unfaded run agrees with the registered Adam fitter. Paired initial coordinates
and losses and every retained best loss are checked. Rerender selected outputs.

One selected-failure set, one prescribed annealing schedule, one trajectory per
method. Not a tuning sweep or representative validation. C04-T0005 and C08-T0000
already informed earlier work; statistics also report the other five separately.
