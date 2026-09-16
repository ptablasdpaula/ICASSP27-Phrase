# Validation of complete takeover on six additional failures

Protocol fixed before running new outcomes, 2026-09-16. The successful development
case C04-T0005 is excluded from these six evaluation cases. No paper changes.

## Case selection

Use the historical ten-target-per-cardinality Fourier-onset pilot, canonical
four-direction CeL. At each of 4,6,8 events, restrict to old onset MAE >25 ms
and exclude the development case. Select (a) the lowest pitch-MAE case among
these timing failures, then (b) the remaining case with highest onset MAE.
Exact IDs and old errors are in [cases.json](cases.json). This selects two strata
of known failures, not a representative random sample of the 150-target set.
Every selected case and result will be retained; no substitution based on outcome.

## Fresh baselines and common conditions

Render targets and rerun registered equal-cell baseline fits at 2.048 padding,
using current canonical four-direction unweighted square-root CeL. Float64 CPU,
fixed excitation amplitudes .8, independent pitch/onset logits, unchanged target
controls. Baseline uses the registered Adam/rollback/stopping implementation.
Report cases that recover during this baseline rather than silently dropping them.

Both interventions restart from the same fresh strict-best baseline controls
with zero moments and LR .05, because the registered fitter does not export its
strict-best moments. Initial event labels are sorted chronologically without
changing physical controls or audio. Trial swaps use adjacent **current onset
ranks** so adjacency does not depend on arbitrary latent labels after crossings.

## Heuristic: freeze the successful pilot settings

One sweep of adjacent pairs 1↔2,...,(N-1)↔N. Keep and swap branch from the same
live state. Minimum 30 updates; patience 40 without relative 1e-4 improvement in
BOTH branches; maximum 300 per branch. Retain strict-best complete branch state.
A swap must beat both live and best keep by 1%. Otherwise adopt keep if it beats
live by 1%, else retain live. Adopt parameters, permutation, moments, count and
LR together. Amplitudes are never learned. Continue from the final adopted
state with the same pilot plateau schedule for at most 3000 additional updates.
No per-case tuning or extra sweeps after inspecting outcomes.

Generalised implementation must first reproduce the development case's three
paired decisions, trial depths and losses to 1e-12. Exact assertions verify
isolation and takeover, and that committed canonical loss never increases.

## Matched ordinary-optimisation control

Allocate the same number of **additional backward updates** as the entire
heuristic (both branches of all trials plus continuation). Ordinary canonical
Adam uses the same initial checkpoint, conditioning scale, initial LR and
continuation/rollback rules. On patience stopping, restart from its best controls
with fresh moments and LR .05 until the budget is exhausted. Keep its strict-best
checkpoint. This is a deterministic restart control, not a random multistart.

Both methods generally render for loss and backward plus checkpoint evaluation
per update. Report actual forward-render counts and wall time as well, since
matched backward counts are not identical wall-clock costs. Baseline cost is
shared and excluded from additional intervention budgets.

## Assessment

Report canonical CeL, jointly Hungarian-matched pitch/onset MAEs, mean normalised
joint event error, and relative waveform L2 as an objective-independent audio
measure. Assignment uses the paper's squared normalised pitch/time cost;
joint event error is the mean Euclidean distance after that assignment. None of
these parameter or independent audio metrics select branches or checkpoints.

Use 20% reduction in joint event error as a descriptive substantial-improvement
benchmark; report recovery below 1 cent AND 1 ms as a stricter benchmark. Compare
against both baseline and compute-matched control, and inspect individual axes
for regressions. These thresholds are pilot reporting conventions. Six selected
failures cannot establish population success rates or justify a general claim
without a wider held-out ablation.

Run `python scripts/validate_takeover_cases.py --qualify`, then `--index 0` through
`--index 5`, with the project environment. Cluster wrapper:
`jobs/takeover_validation.sh`. Raw stage files support resume and reject mismatched
source signatures. Results pending.
