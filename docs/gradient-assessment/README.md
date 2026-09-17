# Full-domain gradient assessment

This experiment compares eleven losses using paired candidate phrases. It does
not change the historical loss registry, recovery runs, or manuscript.

Start with the [findings and limitations](findings.md), [figure](gradient-assessment.pdf),
[PNG preview](gradient-assessment.png), and [full statistics](summary.csv).
The same samples rescored using a positive dot product are available in the
[dot-product comparison](dot-product/README.md).

## Design

The seven columns are: one event with joint displacement; two/four events with
pitch displacement; two/four events with onset displacement; and two/four events
with joint displacement. The single-event target is 160 Hz at 1 s. The two- and
four-event columns each share 32 target phrases, independently generated from the
final recovery registry. Target pitches are uniform in log-frequency from 80 to
320 Hz. Target onsets are uniform in 0.2–1.8 s, conditioned on a minimum 50 ms
separation, then sorted with their associated pitches. They are not isochronous.

Each target initially receives 256 randomised Latin hypercube candidates. Each
unknown pitch is uniform in octaves over 80–320 Hz, and each unknown onset is
uniform in seconds over 0.2–1.8 s. Candidate timing is unrestricted. Conditional
cases project the same joint LHS onto pitch or time and retain the target values
of the other parameter. This gives 49,408 primary candidate–target comparisons
at the initial sample count. No optimisation trajectories are run.

The master seed is 7942029. Target-specific seeds are the first four little-endian
bytes of SHA256 of the tuple containing the master seed, purpose, target ID, and
(for candidates) sample count and repeat index. Increasing sample count generates
a fresh LHS rather than treating a prefix as a smaller Latin hypercube.

## Matching and score

Every candidate uses unrestricted Hungarian matching, identical across losses:

```
cost(i,j) = log2(candidate_pitch[i] / target_pitch[j])**2
            + (candidate_onset[i] - target_onset[j])**2
```

Time is in seconds: a one-second error and a one-octave error have equal cost.
Gradients are with respect to `(log2(f/80), onset_seconds)`, not bounded logits or
range-normalised coordinates. Assignments are held fixed for scoring.

An event succeeds only if the negative gradient points toward its assigned
target in every displaced component. Coordinates within 1e-12 of the matched
target are omitted; zero gradients fail on displaced coordinates. Fully matched
events are omitted. Assignments whose second-best total cost is within 1e-12 of
the optimum are excluded and counted. Nonfinite gradients fail the run rather
than being silently dropped.

**Column labels describe candidate generation.** Unrestricted assignment can
change event correspondence even in a conditional case, creating a matched onset
error where the original onset was correct, or vice versa. The primary score
then checks both displaced components. All coordinates are differentiated even
in conditional cases, allowing this full-descent assessment and measurement of
drift from originally correct parameters.

Scores first average eligible events within a candidate, then candidates within
a target, then targets equally. Supplementary outputs retain separate pitch/time
rates, positive-dot-product rates, cosine alignment, complete-phrase success,
zero gradients and drift from originally correct coordinates. Target-level SD
is descriptive variability, not a confidence interval.

## Loss definitions

Existing settings are preserved:

- L1: waveform mean absolute error. L2: waveform mean squared error, as in the
  existing registry; square-rooting it would preserve nonzero gradient signs.
- Single STFT: mean absolute magnitude difference with periodic Hann-256,
  hop 64, no centring, matching the CeL spectrogram geometry.
- Linear MSS: sum of mean absolute magnitude differences over periodic Hann
  windows 512/256/128/64/32/16, quarter-window hops and centred reflection padding.
  This is SOT's linear MSS component without its 0.05 coefficient.
- Smooth MSS and SOT: unchanged historical implementations. SOT includes the
  published asymmetric transport term plus 0.05 Linear MSS.
- TFW2: linear-Hz transport geometry, 1 s = 1,000 Hz. Log-TFW2: log-frequency
  geometry, 1 s = 1 octave. Both retain their existing transport implementations.
- CeL: all four diagonal cumulative directions, square-root features,
  target-power normalisation and uniform RMS.
- Log-CeL: the same surfaces with the existing Log-Weighing quadrature.
- Fade-CeL: four diagonal surfaces with separable fixed fading, 1 s / 1,000 Hz,
  kernel `max(0, 1-log1p(9*d/H)/log(10))`, and no Log-Weighing or fade schedule.

The matching geometry is common to every loss; it does not change the geometry
or tuning of the losses themselves.

## Efficiency and validation

Target spectra, cumulative references, transport geometry and fading matrices
are cached. Each candidate batch is synthesised once. Identical STFTs and
cumulative surfaces are shared, and SOT reuses Linear MSS values and gradients.
Each loss still receives its own reverse-mode derivative; objectives are never
summed to produce a mixed gradient.

The initial CPU float64 implementation was checked against independent historical paths at
one, two and four events, including near-target candidates, reversed event order
and batch-versus-individual evaluation. Value/gradient tolerances are
`rtol=1e-9, atol=1e-11`. The onset FFT has 16,384 points for 8,000 waveform samples
(2.048x). The existing detached event ordering and hard-reset boundaries are
unchanged: these local derivatives do not certify finite-step convergence across
discrete boundaries.

The campaign runs on GPU in float64. Batch sizes 8/16/32/64 are benchmarked on the
execution node, including all eleven backwards; tuning extends to 128/256 when
the measured memory footprint allows. The fastest batch below 65% of
GPU memory is selected. One GPU worker batches candidates and reuses target
caches, with atomic, source-hashed checkpoints per target and condition. Only
small loss/parameter-gradient arrays are copied back to the host. No audio or
autograd graphs are archived. CPU/GPU equivalence checking is omitted at the
user's request; GPU benchmarks check finite values and gradients. The CPU
implementation and CPU job script remain available as an explicit fallback.
CUDA reflection-padding backward, used by the existing MSS/SOT objectives, is
not guaranteed bitwise deterministic; PyTorch's configured warning is retained.

## Sampling sensitivity

Repeat the centred case and the first eight targets of each multi-event
cardinality with independent LHS seeds. Compare original and repeated mean scores
on the same target subset. If any loss changes by more than two percentage points
for a condition, increase that condition to 512 candidates, then 1,024 if needed.
Keep all losses paired at every stage. Conditions with different final counts
use different LHS designs; pairing between conditional and joint columns applies
within each common sample-count stage. Preserve previous stages for audit.

This repeat check is a sensitivity diagnostic, not a formal confidence interval.
Residual changes above the threshold at 1,024 samples must be reported. The
single-event column characterises only its centred target. Multi-event results
estimate average gradient performance under the specified target/candidate
distributions; neither finite sampling nor positive local gradients establish
an everywhere guarantee or eventual recovery.

## Reproduction

From the repository root:

```bash
export PYTHONHOME="$PWD/.pixi/envs/default"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.pixi/envs/default/bin/python -m pytest tests/test_gradient_assessment.py
sbatch --account=pilot_andrena --partition=andrena --gres=gpu:1 \
  --cpus-per-task=4 --mem=24G --time=04:00:00 \
  --output=results/gradient-assessment-gpu/logs/%j.log \
  jobs/gradient_assessment_gpu.sh
```

The job benchmarks, runs the primary and sensitivity campaigns, and generates
the report. `scripts/assess_gradients.py report --root results/gradient-assessment-gpu`
regenerates the figure and CSVs from completed checkpoints without recomputing gradients.
