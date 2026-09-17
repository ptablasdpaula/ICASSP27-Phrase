# Full 150-per-cardinality recovery study

Six requested configurations: four orthogonal directions averaged uniformly,
the same with Log-Weighing, clockwise rotation without Log-Weighing, and clockwise
rotation with Log-Weighing, fixed 1-second/1000-Hz faded diagonals without
Log-Weighing, and the same faded diagonals with Log-Weighing. The frozen registry contains 150 targets for each of
1, 2, 4, 6 and 8 events: **750 targets × 6 configurations = 4,500 fits**.
All start from the paper's 160-Hz, equal-cell-onset initialisation. No selected
failure subset, amplitude fitting or target-dependent stopping/selection.

## Static orthogonals

Inclusive cumulative sums ↑ ↓ within each frame and → ← within each frequency
bin, averaged equally. Global total target power, square-root feature, no local
mass normalisation. Same Log-Weighing extension as the 510-configuration screen:
oriented quadrature along the accumulated axis, symmetric trapezoidal widths
along the unaccumulated axis. Uniform and LW variants each use their own initial
loss for conditioning.

Use the registered Adam procedure: LR .05, relative improvement threshold 1e-4,
rollback/LR ×.3/moment clearing at patience 100 and 200, stop at patience 250 or
3,000 updates. Preserve the registered implementation's step-counter behaviour.
Retain the strict-best loss iterate. The isolated fitter only adds objective
injection; qualification checks exact agreement with the production fitter.

## Single-stage clockwise recovery

Start from the same standard initial candidate and rotate throughout optimisation:
↗ → ↘ ↓ ↙ ← ↖ ↑, linearly interpolating adjacent directional losses over
100 updates per sector (800 updates per full turn). Update zero uses pure ↗.
There is no diagonal pre-fit, warm-up, refinement or restart stage. Both uniform
and Log-Weighing versions use this one setup.

Use the same registered Adam, learning-rate reductions, rollback, patience and
3,000-update cap as the orthogonal fits. Condition the gradient by the initial
rotating training loss. Advance rotation only on actual Adam updates; rollback
neither advances nor resets its phase.

Because the rotating objective changes with phase, its values are not directly
comparable across updates. The fixed equal mean of all eight directional terms,
with the matching weighting, controls meaningful improvement, patience, rollback
and strict-best checkpoint selection. Only the rotating objective supplies the
training gradient. Orthogonal fits monitor their static four-term objective.
A clockwise run can stop before a full revolution: there is no minimum-cycle
exemption from the common patience rule. Report actual updates and wall time;
all methods share the same maximum budget but may stop at different times.

The earlier staged clockwise jobs were cancelled, and their outputs archived
under `results/direction-recovery-150/superseded-plateau-clockwise`. They are
excluded from this study. Valid completed orthogonal fits are reused unchanged.

## Fixed logarithmically faded diagonals

Two additional variants use all four diagonals with fixed horizons H_t=1 second
and H_f=1000 Hz. Each contribution is multiplied by w(d_t;H_t)*w(d_f;H_f), where
w(d;H)=max(0,1-log(1+9d/H)/log(10)). These are fade-to-zero horizons, not half-weight
horizons. The kernel is logarithmic in both variants; Log-Weighing is a separate
choice of final RMS quadrature. The accumulation uses linear-Hz displacement,
not octave displacement. The LW version uses the original oriented quadrature
in seconds and log-frequency, with the existing 20-Hz coordinate floor.

Reuse exactly the earlier uniform 1s/1000Hz implementation; LW applies existing
per-direction quadrature weights to the same faded feature errors. Global ORIGINAL
target power normalisation, square-root feature, no per-frame mass rescaling.
The fading remains constant throughout: no annealing, delayed patience, warm-up,
plateau restart or refinement. Standard fresh initialisation; registered Adam
LR .05, relative meaningful improvement 1e-4, rollback/clear moments/LR x.3 at
patience 100/200, stop 250 or 3,000 updates, gradient scaling by initial own loss.
Select strict-best OWN fixed training loss for both faded variants.

Qualification checks exact uniform values/gradients/short trajectory against
the pilot, independent oriented LW quadrature, both audio finite differences,
both self-loss/gradient checks, and current shared optimiser/LSD qualification.
The four one/eight-event gate fits cover both weightings before launching 1,500 fits.
Existing orthogonal/clockwise runs and their source signatures remain unchanged.

## Rendering, metrics and provenance

2.048 onset padding (FFT 16,384), fixed .8 amplitudes, independent bounded logits,
CPU float64, compiled TorchLPC, one thread, deterministic algorithms. Identical
750 target coordinates for all configurations. No paper edits.

Hungarian matching uses the registered squared normalised pitch/onset cost.
Per-phrase pitch MAE (cents), onset MAE (ms), and joint RMS normalised event error.
The latter is sqrt(mean squared matched distance), as in the full paper study;
the earlier escape pilots additionally used mean Euclidean distance.

LSD exactly follows the original paper metric: centered periodic-Hann STFT-256,
hop 64, amplitude floor max(target STFT magnitude) ×1e-5 shared by both signals,
and RMS of 20-log10 magnitude differences. It is distinct from the center=False
STFT used by the training losses. Qualification executes the original metric
function to check exact agreement.

Save all fit trajectories, selected metrics, parameters and costs as
compressed per-fit JSON, with source/registry signatures. Resume only matching
signatures. Aggregate only after all 4,500 unique fits validate; report means,
sample SDs and medians by event count/configuration, plus paired method comparisons
with matching weighting and the faded LW-versus-uniform comparison. Raw data remain in the results directory;
commit the full per-phrase table, summaries, plots and archive manifest.

The earlier sensitivity audit applies: single trajectories can respond strongly
to floating-point perturbations. This campaign does not add multiple random
restarts or a period/direction search.
