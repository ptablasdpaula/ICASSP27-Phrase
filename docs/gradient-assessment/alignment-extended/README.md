# Extended gradient alignment study

The GPU extension completed successfully as Slurm job 27287265 on gpushort
(node sbg2) in 15 minutes 34 seconds. All reports and raw archives are available.
The queued andrena job 27284248 was cancelled before starting.

- [Whole-phrase cosine ± SD](phrase-cosine.png) ([PDF](phrase-cosine.pdf), [table](phrase-cosine.md)).
- [Whole-phrase descent percentage](phrase-descent.png).
- [Mean event cosine ± SD](event-cosine.png).

Column order: Both 1, 2, 4, 6, 8; Pitch 1, 2, 4; Time 1, 2, 4.
The seven original columns reuse their frozen primary samples without changes.
The new pitch-only and time-only columns use the existing central one-event
target (160 Hz at 1 s), with their candidates projected from the same seeded
joint LHS generator. The six/eight-event joint columns each use 32 new seeded
targets under the original protocol: uniform log2 pitch from 80–320 Hz and
uniform onset from 0.2–1.8 s, conditioned on 50 ms minimum target separation.
The domain is fixed, so larger cardinalities represent more crowded phrases.

New columns start at 256 candidates per target, with an independent LHS repeat
for the first eight targets (the sole target for single-event columns). As in
the original campaign, the original componentwise percentage criterion selects
sample size: if any loss differs by more than 2 percentage points between repeat
means, increase to 512 and then at most 1024. Phrase-cosine repeat differences
are also recorded, but do not change this frozen selection rule. All eleven
losses are computed on CUDA float64 from a shared synthesis/feature pipeline,
using batches of 64 to allow room for the higher event counts. No CPU/GPU
comparison is performed, as requested.

Primary measure: whole-phrase cosine, with candidate-phrase sample SD. Event
cosine and whole-phrase descent percentage are also exported, using the same
[definitions and exclusions](../alignment/README.md). Hungarian matching uses
squared octave–second distance, with one octave weighted equally to one second.
The full eleven-column preview is 4.9 inches wide to retain readable means and
SDs; its incorporation into the single-column paper is a separate layout choice.

When complete, the output files are phrase-cosine, event-cosine and phrase-descent
(each PNG/PDF/CSV/Markdown), summary.csv, quality.csv and provenance.json.
New raw shards, sampling checks, target design and execution metadata are saved
under extension-data. Original raw shards remain in the parent raw archive;
extract each archive into its recorded extract_into directory to reproduce.
The manuscript and original seven-column reports remain unchanged.

```bash
sbatch --account=pilot --partition=gpushort --gres=gpu:1 \
  --cpus-per-task=4 --mem=32G --time=01:00:00 \
  --job-name=cel-gradient-extra \
  --output=results/gradient-assessment-extended/logs/%j.log \
  jobs/gradient_cardinality_gpu.sh
```

The job resumes existing validated shards. It then runs the reporting and
archiving commands automatically; failures stop the job and appear in its log.

## Results and validation

The combined report contains 84,480 primary candidate–target comparisons. New
joint columns use 256 candidates for each of 32 targets; single-event pitch uses
512 candidates and time uses 1024. Original seven-column values, including SDs,
match the previous report exactly. All raw-file and archive hashes were verified.
The eleven-column phrase-cosine figure was visually inspected; no clipping or
label overlap was found.

For joint displacement, Fade-CeL has higher mean whole-phrase cosine than CeL
at six events (0.16 versus 0.14) and eight events (0.12 versus 0.10).
Whole-phrase descent percentages are 73.1% versus 69.9% at six events and
68.6% versus 66.2% at eight. This supports a modest advantage for Fade in the
more crowded sampled phrases, not a large improvement or a convergence claim.
Alignment declines substantially with cardinality for all losses. No statistical
significance test has been performed.

The single-event time column reached the 1024-candidate cap without satisfying
the 2-percentage-point repeat threshold for all losses. Repeat differences in
mean phrase cosine reached 0.084 for Single STFT, 0.047 for SOT and 0.048 for
log-TFW2; all CeL variants differed by less than 0.0001. Thus the displayed
single-event baseline values have visible sampling sensitivity even though the
CeL results are stable. See extension-data/sampling.json for every check.

## Combined presentation

[Mean phrase cosine with positive-phrase percentages](phrase-cosine-positive.png)
([PDF](phrase-cosine-positive.pdf), [Markdown](phrase-cosine-positive.md),
[CSV](phrase-cosine-positive.csv)) places the mean cosine above the percentage
of candidates with a positive summed phrase dot product, shown in parentheses.
Colour encodes cosine only. Cosines omit the leading zero (including negatives,
e.g. −.01); values rounding to zero display .00. This is a presentation of the
existing summary.csv results, with no new gradients or changed scores.
Reproduce with `scripts/render_gradient_cosine_positive.py` using the project Python.
