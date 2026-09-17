# Execution and validation

All seven Slurm array tasks in job 27186594 completed successfully, each running
clockwise, fixed-eight and diagonal-only variants sequentially on one CPU node.
Individual tasks took 7m12s–9m47s; the array ran in two waves with concurrency four.
CPU nodes by index 0–6: ddy31, ddy56, ddy98, ddy161, ddy19, ddy31, ddy116.
Float64, PyTorch 2.7.1+cu126, deterministic algorithms, one Torch thread and the
compiled TorchLPC backend. No campaign jobs remain active.

21 runs × 2,400 updates = 50,400 backward updates. Saved trajectories include all
2,401 evaluated states per run, directional weights/losses, physical coordinates,
and periodic matched errors. Selected, exploration-end and final states are
reported separately. Raw per-run source and baseline hashes are preserved.

Checks passed:

- Rotation follows ↗ → ↘ ↓ ↙ ← ↖ ↑ in the eight-direction implementation order.
  Tested every weight vector over 2,401 steps for nonnegativity/unit sum, the eight
  pure-direction positions and sector midpoints, and equal full-cycle weights.
- Every baseline re-render agrees with the archived canonical loss within 1e-12.
- Every actual backward pass checks finite gradients.
- All 21 final selections equal the minimum common loss over the run and incumbent
  within 1e-12, with exactly 2,400 updates and 2,401 logged states each.
- All logged weights are nonnegative and sum to one; best-update indices reproduce
  the reported final loss. `checkpoints.json` records these indices and raw hashes.
- Changed Python files pass Ruff. The comparison plot was visually inspected.
- `raw-results.zip` contains the 21 complete JSON trajectories; its integrity was checked.

The existing orthogonal losses are reused from the previous eight-direction screen,
where explicit sums, quadrature and mixed-gradient finite differences were tested.
No synthesis, production loss, paper, amplitude-fitting or target-selection changes
were made for this pilot. Compute protocol/source: commit `32f4482`, with the exact
script hash in every result file.
