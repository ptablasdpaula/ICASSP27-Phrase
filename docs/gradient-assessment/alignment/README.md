# Gradient alignment tables

These three paired tables reuse the final primary (`r0`) GPU campaign samples: 66,560
candidate–target comparisons, eleven losses and seven conditions. No synthesis,
new gradients, or sample selection was performed for these metrics. The original
componentwise and per-event positive-dot reports remain unchanged.

- [Mean event cosine ± SD](event-cosine.md) ([PNG](event-cosine.png), [PDF](event-cosine.pdf), [CSV](event-cosine.csv)).
- [Whole-phrase descent percentage](phrase-descent.md) ([PNG](phrase-descent.png), [PDF](phrase-descent.pdf), [CSV](phrase-descent.csv)).
- [Mean whole-phrase cosine ± SD](phrase-cosine.md) ([PNG](phrase-cosine.png), [PDF](phrase-cosine.pdf), [CSV](phrase-cosine.csv)).

Let d_i be negative loss gradient for event i and r_i its displacement towards
its unrestricted Hungarian-matched target. Matching minimises squared distance
in octave–second coordinates, with 1 octave weighted equally to 1 second.
This is a chosen parameter metric. It measures note-event recovery independently
of synthesiser event indices; it is not a perceptual audio metric.

1. Event cosine: average d_i·r_i/(||d_i|| ||r_i||) over displaced events within
   each candidate phrase, then average these phrase scores across candidates.
2. Whole-phrase descent: percentage of candidates with sum_i d_i·r_i > 0.
   This is the local descent criterion for total matched squared parameter error,
   allowing improvements in one event to outweigh deterioration in another.
3. Whole-phrase cosine: (sum_i d_i·r_i) /
   (sqrt(sum_i ||d_i||²) sqrt(sum_i ||r_i||²)). This concatenates the complete
   phrase vectors; it does not add individually normalised cosines.

All gradient components are retained, including gradients on already-correct
coordinates and events. Their motion reduces whole-phrase alignment through its
denominator. Conditional columns indicate which coordinates were initially
perturbed; unrestricted reassignment can introduce errors in both coordinates.

SD is the sample standard deviation (ddof=1) across candidate phrase scores,
including both within-target and between-target variability. It is descriptive,
not a confidence interval. Equal candidate counts per target and no exclusions
in this campaign give equal target weighting. CSV means for phrase descent are
fractions; displayed tables and colour scales use percentages. CSV also records
candidate SD of the binary descent indicator, although only cosine SD is plotted.

Zero descent has cosine 0 by convention and does not pass the positive-dot test.
Fully matched events are excluded from the event-cosine average. Fully matched
phrases and assignment ties are excluded from all three summaries, following the
existing 1e-12 tolerance; nonfinite gradients cause an error. Quality counts and
raw-file hashes are provided in quality.csv and provenance.json.

Validation checks archived mean event cosines, equality of the two single-event
cosines, and equality of the signs of phrase cosine and summed dot. Unit tests
cover cancellation between events, zero gradients, exact matches, ties,
already-correct-coordinate drift, permutation invariance and nonfinite inputs.

These are local diagnostics within the sampled domain, not guarantees of
convergence or coverage of the entire continuous gradient space. Sampling sizes
were chosen using the original componentwise criterion; they have not been
retuned for these new metrics. See the [parent protocol](../README.md) for design
and sampling sensitivity, including the single-event sensitivity cap.

Reproduce from the repository root:

```bash
PYTHONHOME="$PWD/.pixi/envs/default" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .pixi/envs/default/bin/python scripts/report_gradient_alignment.py
```

The manuscript is unchanged.

## Findings

The CeL family has the highest mean event cosine, mean whole-phrase cosine and
whole-phrase descent percentage in every tested column. This is descriptive;
no statistical significance test is claimed.

For four-event joint displacement, whole-phrase descent is 79.1% for CeL, 79.9%
for Log-CeL and 80.3% for Fade-CeL, versus 63.5% for TFW2. Their whole-phrase
cosines are respectively 0.25 ± 0.32, 0.26 ± 0.32 and 0.26 ± 0.32, versus
0.11 ± 0.33. Thus a favourable sign does not imply a nearly direct path to the
matched target. Alignment weakens as phrase cardinality increases.

Fade is not uniformly better: for two-event joint displacement its whole-phrase
cosine is 0.40 ± 0.40, versus 0.44 ± 0.40 for CeL. It improves pitch-only cosine,
but does not uniformly improve whole-phrase descent percentages. Log-Weighing
produces comparatively small increases over CeL in these diagnostics.

SOT illustrates why sign and cosine complement one another: its two-event
pitch-only whole-phrase descent is 77.4%, yet its mean whole-phrase cosine is
−0.01. A majority of positive dots can coexist with a slightly negative average
cosine when the negative alignments are larger in magnitude.
