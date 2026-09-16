# Diagonal refinement from the last orthogonal iterate

Follow-up to [the six-phrase switch experiment](../orthogonal-validation/README.md).
The previous procedure selected the **best original diagonal-loss checkpoint**
from the orthogonal phase, then refined that checkpoint. This experiment instead
starts from the **actual final orthogonal iterate at update 1600**, even if its
diagonal loss is worse than the original stalled state or an earlier iterate.

Test both variants (four orthogonals and frequency-only) on all six additional
phrases, with no new selection of cases or exploration reruns. Use the exact
saved final pitches and independent onsets. Amplitudes remain fixed at 0.8,
padding at 2.048, and the objective switches completely back to the four-direction
diagonal CeL. Start fresh Adam with the same registered settings as the earlier
refinement: LR .05, objective divided by its initial value, maximum 3000 updates,
plateau patience 100 and stop patience 250. Optimiser moments are reset;
this continues the final **parameters**, not the old orthogonal Adam moments.

Retain the best diagonal loss encountered within this new refinement only.
There is no fallback to the original stalled fit or earlier orthogonal
checkpoints. Compare the last orthogonal state, its new diagonal refinement,
and the previous best-checkpoint refinement. Target parameter errors never
select controls. Report both improvements and regressions relative to the
previous procedure, and retain exact update counts. Maximum refinement budgets
match, but actual early-stopping lengths can differ. The previous diagonal-only
controls are contextual references and are not newly budget-matched to these
runs; this is a restart-state diagnostic, not a fresh compute-matched benchmark.

Implementation: `python scripts/refine_last_orthogonal.py --indices 0 1 2 3 4 5`.
Raw outputs include full refinement trajectories, starting controls, source-file
SHA256, target/renderer/optimizer provenance and previous results. Verify the
starting diagonal loss against the saved update-1600 loss within 1e-12 and verify
fixed amplitudes throughout. Production code and the paper are unchanged.

Results pending.
