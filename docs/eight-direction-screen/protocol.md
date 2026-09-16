# Complete diagonal + orthogonal subset screen

Requested extension of the [original gradient screen](../cel-gradient-screen/protocol.md).
Same seed 2028, 173 targets, 13,099 candidates, event counts 1/2/4/6/8,
structured pitch-only, onset-only and joint slices, independent simultaneous
errors and original fit initialisations. No recovery runs or variant selection.

Direction bits: 1 ↗, 2 ↘, 4 ↖, 8 ↙, 16 ↑, 32 ↓, 64 →, 128 ←.
Every nonempty subset of the eight directions is included: 255 subsets,
each without and with Log-Weighing, hence 510 configurations. This includes
all orthogonal-only subsets and all mixed diagonal/orthogonal subsets.
Each objective is the equally weighted mean of its selected directional RMS
terms, with no separate scaling of the two families.

Orthogonal ↑/↓ sum only frequency bins independently within each frame;
→/← sum only time frames independently within each frequency bin. As in the
recent escape pilot, all use global total target power, not per-frame/per-bin
normalisation. STFT, square-root feature/floor, fixed amplitudes, 2.048 onset
padding, and all synthesis settings are unchanged.

For Log-Weighing, the accumulated axis uses the existing orientation-dependent
quadrature widths. The axis without accumulation has no orientation: use
trapezoidal widths (the mean of forward and backward widths), in log2-frequency
and linear time respectively. Thus up/down LW weights average the corresponding
forward/backward-time diagonal quadrature weights; right/left average the two
frequency orientations. This is an explicit extension of the earlier quadrature
rule, not a log-STFT or local mass normalisation.

CPU float64 with deterministic algorithms and compiled TorchLPC. Reuse archived
diagonal elementary losses/gradients only after validating source hashes,
targets, candidates and perturbation labels. Recompute all eight diagonal
values/gradients on the first batch of every target as an independent agreement
check. Compute the eight new elementary terms (four ordinary/four LW), then
reconstruct all 510 subset gradients by linearity. Save all 16 elementary
losses/gradients and the original candidate/matching arrays in each shard.

Scoring is unchanged: descent alignment with matched target displacement in
normalised physical coordinates; pitch and onset components, joint cosine/sign,
whole-phrase direction, exact-zero gradients and drift of already-correct
coordinates. Exclude tied assignments, exclude correct coordinates from axis
alignment denominators. Average candidates within target/type, then equally
weight targets and report mean/sample SD/median. Also report paired differences
from the uniform four-diagonal reference. Reproduce every archived diagonal-only
summary statistic before accepting the extended report.

The structured timing-only slices may acquire pitch displacement through
Hungarian reassignment, as before. Positive local alignment does not establish
finite-step improvement or convergence. With 510 configurations, apparent best
subsets are exploratory and cannot be treated as validated winners.

Run `bash jobs/eight_direction_screen.sh`, then
`python scripts/report_eight_directions.py` in the pinned environment.

See [execution and validation](execution.md) for the completed run and checks.
