# Gradient assessment: results and limits

The [single-column figure](gradient-assessment.pdf) shows the percentage of events
whose descent points toward the Hungarian-matched target in every displaced
component. [PNG preview](gradient-assessment.png), [full statistics](summary.csv),
and [target-level scores](target-scores.csv) are also available. CSV rates are
fractions; the figure and table below display percentages.

The seven primary columns contain **66,560 candidate–target comparisons** after
sample-count refinement. The initial 256-candidate screen is preserved separately
in [initial/](initial/gradient-assessment.pdf). All stored stages and independent
repeats contain **105,984 comparisons**, each evaluated with eleven losses.

| Loss | Both, 1 | Pitch, 2 | Pitch, 4 | Time, 2 | Time, 4 | Both, 2 | Both, 4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| L1 | 23.6 | 53.3 | 42.2 | 45.4 | 40.8 | 25.7 | 25.1 |
| L2 | 16.0 | 50.7 | 39.7 | 45.6 | 42.1 | 23.5 | 25.5 |
| Single STFT | 33.5 | 60.4 | 48.6 | 59.8 | 48.2 | 33.4 | 31.4 |
| Linear MSS | 28.7 | 63.5 | 50.4 | 51.5 | 45.9 | 33.0 | 32.0 |
| Smooth MSS | 30.3 | 71.1 | 54.9 | 49.8 | 44.8 | 34.1 | 32.4 |
| SOT | 24.9 | 69.1 | 55.9 | 45.4 | 40.1 | 26.8 | 26.4 |
| TFW2 | 41.0 | 63.1 | 49.7 | 59.1 | 46.7 | 35.7 | 32.4 |
| log-TFW2 | 47.9 | 65.3 | 52.3 | 54.0 | 44.8 | 38.2 | 34.0 |
| CeL | 84.4 | 76.8 | 60.8 | 73.5 | 55.3 | 56.9 | 46.1 |
| Log-CeL | 84.4 | 77.8 | 61.7 | 73.8 | 55.6 | 57.1 | 46.7 |
| Fade-CeL | 73.9 | 83.2 | 65.9 | 74.2 | 56.8 | 56.0 | 47.8 |
| Candidates per target | 1,024 | 512 | 256 | 256 | 256 | 512 | 256 |
| Targets | 1 | 32 | 32 | 32 | 32 | 32 | 32 |

## Interpretation

- All three CeL variants have higher sampled mean success than every non-CeL
  comparator in every column. This is a descriptive ordering, not a significance
  test or a claim about convergence.
- Fade improves over ordinary CeL by approximately 6.4 and 5.1 percentage points
  in the two-/four-event pitch-displaced columns. Its joint-displacement changes
  are mixed: about -0.9 points with two events and +1.7 with four. It is weaker
  for the centred single-event target.
- Log-Weighing changes the multi-event CeL means by less than one percentage
  point. These small differences should not be treated as established superiority.
- Log-TFW2 improves the pitch-displaced and joint means over linear TFW2, while
  reducing its onset-displaced means. This is a trade-off between conditions.
- Smooth MSS and SOT are more competitive in pitch-displaced cases than in
  onset-displaced or joint cases. These results do not imply that SOT is
  mathematically invariant to timing.

## Sampling sensitivity

The largest absolute change across eleven losses in the independent-seed check
at each column's final sample count is:

| Column | Maximum change (percentage points) | Two-point threshold |
|---|---:|---|
| Both, 1 | 3.906 | Not met at the 1,024-candidate cap |
| Pitch, 2 | 0.952 | Met |
| Pitch, 4 | 1.086 | Met |
| Time, 2 | 1.733 | Met |
| Time, 4 | 0.940 | Met |
| Both, 2 | 1.538 | Met |
| Both, 4 | 0.867 | Met |

For the centred single-event case, Single STFT, Linear MSS and Smooth MSS change
by 3.809, 2.051 and 3.906 points respectively. CeL, Log-CeL and Fade-CeL change by
0.098, 0.391 and 0.098 points. This residual sensitivity is retained explicitly;
the first registered sample remains the primary result. The repeat is not pooled
into it, and no further sampling beyond the predeclared cap was performed.

Multi-event checks average the same first eight target phrases for each seed.
These checks assess sampling sensitivity, not confidence intervals. Target-level
SD and medians in the CSV describe between-target variation separately. The
single-event column has one target and therefore no between-target SD.

## Unrestricted matching changes conditional interpretation

Matching minimises squared octave error plus squared second error in every
column. The fraction of candidates with at least one reassigned event is 25.8%
and 75.6% for two-/four-event pitch displacement, and 20.2% and 68.9% for onset
displacement. Consequently, column labels describe how candidates were generated;
after matching, both components may be displaced and both are scored.

The joint-column reassignment rates are 50.0% and 95.4%. Joint candidates have
arbitrary event labels, so these high rates are not themselves failures.

## Execution and reproducibility

Slurm job **27248959** completed on an NVIDIA A100-PCIE-40GB in **27 min 13 s**,
including GPU batch benchmarking, all saved sampling stages and figure generation.
Batch 256 was fastest among 8/16/32/64/128/256 and peaked at approximately 14.6 GiB
allocated GPU memory in the benchmark. The saved runs contain no nonfinite
gradients and no ambiguous assignments.

The six new design/scoring/checkpoint tests passed. Shared arithmetic was checked
against independent historical loss implementations on CPU before the CUDA port.
CPU/GPU equivalence checking was omitted at the user's request. CUDA reflection-
padding backward is not guaranteed bitwise deterministic; this is recorded in
[execution.json](execution.json).

All 326 shards, including superseded stages and independent repeats, are archived
in [raw/](raw/), split into 15 files totalling approximately 59.5 MB. Archive and
shard hashes are in [raw-manifest.json](raw-manifest.json). Extract the archives
into `results/gradient-assessment-gpu` and copy `design.json`, `sampling.json` and
`qualification.json` there to regenerate the report without rerendering audio.

The manuscript has not been changed by this experiment.

## Suggested figure caption

Percentage of target-directed event gradients. Headers indicate initially
displaced parameters and event count. Each gradient must point toward its target
in every displaced component after unrestricted Hungarian matching, with 1 s
and 1 octave assigned equal cost. Column labels describe candidate generation;
matching may introduce displacement in either component. The single-event target
is 160 Hz at 1 s; other columns average 32 randomly pitched and timed targets.
Candidate counts are 1,024/512/256/256/256/512/256, left to right. Independent-seed
changes are below two percentage points for all multi-event columns and reach
3.9 points for the single-event column.
