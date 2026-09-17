# Full 150-per-cardinality recovery study

Four requested configurations: four orthogonal directions averaged uniformly,
the same with Log-Weighing, clockwise rotation without Log-Weighing, and clockwise
rotation with Log-Weighing. The frozen registry contains 150 targets for each of
1, 2, 4, 6 and 8 events: **750 targets × 4 configurations = 3,000 fits**.
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
signatures. Aggregate only after all 3,000 unique fits validate; report means,
sample SDs and medians by event count/configuration, plus paired clockwise-versus-orthogonal comparisons
with matching weighting. Raw data remain in the results directory;
commit the full per-phrase table, summaries, plots and archive manifest.

The earlier sensitivity audit applies: single trajectories can respond strongly
to floating-point perturbations. This campaign does not add multiple random
restarts or a period/direction search.
