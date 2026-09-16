# CeL gradient screen protocol

This screen precedes variant selection and optimisation. It changes onset FFT
length to 16384 (2.048 × 8000 samples); target and candidate audio are rendered
with that same setting. Other synthesis parameters, the square-root feature,
STFT and target-power normalisation are unchanged. The earlier padding audit
supports reduced wrap-around error, not exact elimination.

The full design has 173 targets and 13099 candidate configurations. At each
of 1, 2, 4, 6 and 8 events, 30 independent targets use NumPy seed 2028:
onsets are sorted uniform draws in 0.2–1.8 s, rejecting draws with less than
50 ms separation; pitches are independent log-uniform draws in 80–320 Hz.
The frozen final-evaluation targets are untouched. Structured targets use
isochronous onsets from 0.4 to 1.6 s (1 s for a single event), repeated pitches
100/160/256 Hz, and ascending/descending log-spaced 100–256 Hz sequences for
multiple events.

For structured targets, each event is independently displaced over the Cartesian
product of offsets 0, ±0.025, ±0.05, ±0.1, ±0.2 in normalised pitch/onset
coordinates. The unperturbed origin and out-of-range coordinates are omitted;
other events remain correct. Timing-only, pitch-only and joint slices are kept
separate. Every target additionally gets 32 full-range scrambled Sobol candidate
phrases (seed 2028 plus the first four SHA256 bytes of its ID, little-endian),
and the existing equal-cell candidate initialisation. Candidate ordering and
spacing are unrestricted, as in the fitting experiment.

`cel_01` through `cel_15` select nonempty direction subsets by bit mask:
1 right/up, 2 right/down, 4 left/up, 8 left/down. Suffix `_lw` applies
Log-Weighing. Their losses and gradients are means of the corresponding
elementary directional terms. The four-direction variants reproduce the legacy
losses. At exact agreement, a vector norm supplies the zero subgradient,
including for Log-Weighing.

Hungarian matching uses squared Euclidean distance in
(log2(f/80)/2, (t−0.2)/1.6). A second assignment within 1e-12 total cost is
considered ambiguous: those candidate phrases are saved and counted but excluded
from primary summaries. Assignments are held fixed for scoring. Coordinates
within 1e-12 of their matched targets are excluded from directional denominators.
Zero gradients fail if displacement is nonzero. Nonfinite gradients are counted
and excluded, never silently replaced. Drift in already-correct coordinates is
the fraction of the full gradient norm lying in those coordinates.

Report per-event pitch/onset sign agreement, joint positive dot product and
joint cosine, plus whole-phrase positive dot product. Event scores are averaged
within candidate phrase, then across candidates of the same target/type, then
summarised across targets by mean, sample standard deviation and median.
Profile and cardinality remain separate. No winner is automatically selected.

These are local derivatives in normalised physical coordinates, not Adam/logit
updates. State-reset boundaries and event ordering remain detached exactly as
in the synthesiser; these scores do not imply finite-step improvement across
a boundary or convergence.

## Reproduction

From the repository root, with the pinned environment/backends installed:

```bash
# On a GPU node: verify CPU/GPU agreement before GPU computation.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .pixi/envs/default/bin/python \
  scripts/screen_cel_gradients.py qualify --output results/cel/qualification.json

# Compute one target; use an array over indices 0–172 for the complete screen.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .pixi/envs/default/bin/python \
  scripts/screen_cel_gradients.py compute --device cuda --index 0 \
  --qualification results/cel/qualification.json --output results/cel/raw

.pixi/envs/default/bin/python scripts/screen_cel_gradients.py report \
  --input results/cel/raw --output docs/cel-gradient-screen
```

`jobs/cel_gradient_screen.sh` accepts repository path, output directory and
`qualify`, `compute`, or `figure`. Submit qualification first; launch the
173-element compute array only after it succeeds. Limit concurrency to four
GPU jobs. Shards are atomic and resumable only for the same source hash/backend.
CPU execution is also supported and is always explicitly labelled.

The new Figure 3 script recomputes all six surfaces and directions without
reusing archived values. Historical caches retain their original padding.
Existing Table I/Figure 4 recovery results are not updated or relabelled by this
screen; final 150-target reruns follow review of the gradient comparison.
