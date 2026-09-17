# Clockwise eight-direction descent pilot

Frozen before inspecting outcomes: development case C04-T0005 and the six
previously selected failed phrases in `docs/takeover-validation/cases.json`.
These are plateau restarts, not fresh-start training or a representative sample.

Three paired runs per case, all starting at the saved diagonal plateau:

- Clockwise rotation: ↗ → ↘ ↓ ↙ ← ↖ ↑, then repeat. Linear interpolation
  between adjacent directions, 100 updates per sector, 800 per full turn.
  During the first 100 updates, linearly ramp from uniform four-diagonal weights
  to the rotating weights. Run 1,600 exploration updates (two turns).
- Uniform eight: equal weights on all eight directions for 1,600 updates.
- Diagonal control: equal weights on the original four diagonals for 1,600 updates.

All then receive 800 diagonal-only refinement updates from the **actual final
exploration state**, even if its diagonal loss is worse than the incumbent.
Use fresh Adam at the start of exploration and refinement. Exploration LR .05;
refinement LR .05, .015, .0045, .00135 for four successive 200-update blocks.
Do not restart or roll back within either phase. All runs use exactly 2,400
backward updates; this matches updates, not wall time (eight-term objectives
cost more per update).

Keep the strict-best canonical four-diagonal loss over the whole trajectory,
including the original plateau as fallback. Also retain final exploration and
refinement states separately. Ground-truth pitch/onset errors never select
checkpoints. This permits endpoint refinement before deciding whether to retain
an excursion, without unconditionally adopting a worse-loss result.

Uniform spatial RMS (no Log-Weighing), global total-target-power normalisation,
square-root feature, fixed amplitudes .8, independent bounded pitch/onset logits,
onset padding 2.048, CPU float64 and compiled TorchLPC. All directions use their
raw elementary losses; no per-direction or per-family rescaling. Divide each
objective by the same initial four-diagonal loss, fixed throughout the run.

One registered period and direction of rotation; no schedule search. Compare
canonical loss, matched pitch/onset errors, joint event error and waveform L2.
A reduction in CeL alone need not improve parameter recovery. Selected plateau
cases and a single schedule cannot establish general training benefits.

Run `scripts/test_clockwise_eight.py --index 0` for the development case and
indices 1–6 for the frozen additional cases. `jobs/clockwise_eight.sh` is the
Slurm array wrapper. See [results](README.md), [interpretation](findings.md), and [validation](execution.md).
