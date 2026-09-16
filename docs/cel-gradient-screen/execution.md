# Execution and validation

The complete screen contains 173 targets and 13,099 candidate phrases, evaluated
using the compiled PhilTorch/TorchLPC **CPU** backend, float64, PyTorch
2.7.1+cu126, and an onset FFT length of 16,384. Zero ambiguous assignments and
zero nonfinite variant/candidate gradients were found. No displaced events had
an exactly zero full gradient. Already-correct coordinates are excluded from
directional denominators as specified in the protocol.

Completed CPU shards from the workspace were reused when moving the run to
Apocrita node `ddy26`. Slurm job 27153044 completed successfully in 10 minutes
29 seconds with four CPU workers and 16 GB requested memory. Small batches
were used after local workers exited with code 137. Atomic
per-target checkpoints preserved completed work. Raw arrays are authoritative;
all source hashes and backend identifiers agree across the 173 shards.

GPU qualification job 27152456 remained queued for priority and was cancelled.
**CPU/GPU agreement was not established for this screen, and no GPU campaign
was run.** The qualification command remains available for a later check.

Validation:

- All 53 repository tests passed, including legacy value/gradient equivalence,
  all direction subsets, explicit rectangle sums, exact-match gradients,
  matching/permutation behaviour and report aggregation.
- The reporting test was rerun after adding isolation of nonfinite directions;
  a failed elementary direction does not contaminate unrelated subsets.
- Changed Python files pass Ruff. Repository-wide lint additionally reports
  four pre-existing import-order warnings in unmodified files.
- The updated paper builds without undefined references; it retains six pages
  and the waveguide at the bottom right of page 2.
- Figure 3 was freshly recomputed on CPU at the new padding. Its counts remain
  72, 59, 51, 62, 91 and 120, in the existing six-objective order. The new
  surfaces, arrows and provenance are committed with the paper.

No gradient-descent pilot or final recovery reruns were performed. Existing
Table I/Figure 4 results retain their original padding and are marked as
awaiting reruns in the paper.
