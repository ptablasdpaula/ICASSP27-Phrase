# Extended gradient alignment study

The GPU extension was submitted as Slurm job 27284248. This directory receives
reports automatically on successful completion. The presence of this README
alone does not indicate that results are available.

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
sbatch --account=pilot_andrena --partition=andrena --gres=gpu:1 \
  --cpus-per-task=4 --mem=32G --time=01:00:00 \
  --job-name=cel-gradient-extra \
  --output=results/gradient-assessment-extended/logs/%j.log \
  jobs/gradient_cardinality_gpu.sh
```

The job resumes existing validated shards. It then runs the reporting and
archiving commands automatically; failures stop the job and appear in its log.
