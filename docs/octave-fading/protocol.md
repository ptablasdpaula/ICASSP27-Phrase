# One-second / one-octave fading pilot

Repeat C08-T0000 with logarithmic fade-to-zero horizons of 1 second and 1 octave.
Two runs: the same saved diagonal plateau and standard fresh initial guess as
[the linear-Hz pilot](../fading-diagonal/protocol.md). Slurm array 27189470.

The kernel is unchanged: w(d; H) = max(0, 1-log(1+9d/H)/log(10)). Time distance
is in seconds; frequency distance is abs(log2(max(f,20)/max(f_source,20))).
The 20-Hz floor matches the existing CeL log-frequency coordinate convention.
Causality still follows bin indices, including the two distinct bins below
20 Hz. A contribution has zero weight at 1 octave or 1 second, and half-weight
at about .2403 octaves or .2403 seconds. Frequency distance is in octaves but
final RMS averaging is still uniform across STFT bins: no Log-Weighing is added.

All four diagonal directions use the same kernel. Because octave coordinates
are nonuniform on the linear STFT grid, downward scans use the transpose of
the causal frequency matrix, not a reflected copy of its forward weights.
Independent explicit rectangle sums verify all four directions, including this
reverse scan, and audio finite differences verify gradients. Self-loss and
self-gradient are zero; the one-octave cutoff is verified.

Everything else follows the previous pilot: original global target power
normalisation, square-root feature, STFT256/hop64/center=False, fixed .8
amplitudes, 2.048 onset padding, independent bounded pitch/onset coordinates,
CPU float64 single-thread deterministic execution, registered Adam .05 and
patience100/200 rollback/LR reduction, stop250 or3000 updates. Each run selects
the strict best own-objective iterate. No refinement or parameter-error-based
checkpoint selection. Existing sources and the full 150-target study are unchanged.

Compare each start with its own earlier no-fade and 1 s / 1000 Hz results.
Raw training losses across kernels are not comparable. Report pitch/onset MAE,
joint RMS matched error, common unfaded loss, LSD, and update counts. This is
one phrase and one run per starting condition, not evidence of generalisation.
