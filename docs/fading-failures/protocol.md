# Comparison across the seven frozen failed phrases

42 fits: C04-T0005 (original development case), C04-T0003, C04-T0007,
C06-T0000, C06-T0008, C08-T0000, C08-T0004; three objectives; two starts.
Slurm array 27189928, maximum21 concurrent single-CPU jobs.

Objectives: four equally averaged diagonals with logarithmic fading:
1 s / 1000 Hz, 1 s / 2 octaves, 1 s / 4 octaves. Fade weight at separation d
and horizon H is max(0, 1-log(1+9d/H)/log(10)). Horizons are zero-weight
distances, not half-weight distances (half-weight is .2403 H). Time and
frequency weights multiply for each contribution inside a directional rectangle.
Octave coordinates are log2(max(f,20)/20); causality follows bin indices.
Reverse frequency scans use transposed weights on the nonuniform octave grid.
Linear-Hz runs reuse the earlier fading implementation without arithmetic edits.

Each objective is run from (a) exact saved diagonal plateau and (b) standard
160-Hz/equal-cell initialisation. All three start identically for each case/mode.
Rerun every fit, including the C08-T0000 linear-Hz comparator; archive the previous
results unchanged. There is no unfaded control in this expanded comparison.
The saved plateau itself is recorded through initial metrics.

Global original target power normalisation; square-root feature and 1e-12 floor;
STFT256 periodic Hann, hop64, center=False; no Log-Weighing; fixed .8 amplitudes;
independent bounded pitch/onset logits; 2.048 onset padding (FFT16384). CPU float64,
compiled TorchLPC, one thread, deterministic algorithms. Same registered Adam
LR .05, own initial-loss conditioning, relative improvement1e-4, rollback/clear
moments/LR x.3 at patience100/200, stop250 or3000 updates. No refinement.
Strict-best own-objective checkpoint selection, never parameter-error selection.

Report per-phrase pitch MAE, onset MAE, joint RMS matched error, LSD, common
unfaded diagonal loss, updates and wall time. For each start, pair octave results
with 1000-Hz results. Aggregate mean/sampleSD/median and recovery counts (<1 cent
AND <1 ms) both for all seven and for the five excluding C04-T0005/C08-T0000.
These two cases already informed development and fade exploration. The other five
are still selected failures, not a representative or untouched benchmark.

Qualification independently checks rectangle sums for all four directions and
both new octave horizons, octave cutoffs, autograd and audio finite differences,
zero self-loss/gradient, and exact reproduction of the existing one-octave route.
Each saved best objective is checked by rerendering. The source signature includes
all reused pilot/shared sources, the registry, case list and saved plateaus.
No existing production or pilot source is changed.

One trajectory per setting. Previously observed sensitivity to round-off and CPU
execution limits robustness claims. Different objectives can stop after different
numbers of updates under the same patience; raw loss magnitudes are not comparable.
