# Findings for review

These comparisons concern local gradients, not optimisation success. No variants have been selected for the recovery experiment.

- The centred Figure 3 case remains 120/120 for the four-direction CeL. Across the 30 independent one-event targets and 32 candidates each, its joint alignment is 98.5%, while pitch alignment alone is 85.2%. A favourable joint direction can conceal a wrong pitch component.
- With onset coordinates correct, the structured one-, two- and four-event pitch-only slices give 100% pitch alignment for both four-direction variants. This contrasts with the simultaneous-error cases and points to timing–pitch interactions in those tests.
- For the four-direction variant, Log-Weighing increases mean pitch alignment at two, four, six and eight events. Its onset changes are mixed. The mean joint changes are small relative to the target-to-target standard deviations below.
- Direction subsets trade pitch against timing. For example, at four events, the downward-frequency pair (↘ ↙) has 72.6% pitch and 65.8% onset alignment, compared with 69.2% and 64.6% for all four directions. At one event, the forward-time pair (↗ ↘) has 94.5% onset alignment versus 92.0% for all four. These descriptive results do not establish statistical superiority or eventual convergence.

In the eight-event ascending/descending timing-only slices, Hungarian reassignment can introduce matched pitch displacement. Correct-coordinate masks follow the matched targets, so those slices also have eligible pitch scores.

## Four-direction variants: joint alignment

Percentages below summarise the target-level rates: each target first averages its 32 candidate configurations, and the table then reports mean ± sample SD and median across 30 targets. These are not confidence intervals.

| Events | Uniform: mean ± SD; median | Log-Weighing: mean ± SD; median |
|---:|---:|---:|
| 1 | 98.54 ± 1.97; 100.00 | 98.44 ± 2.13; 100.00 |
| 2 | 83.91 ± 8.31; 85.16 | 83.75 ± 8.99; 85.94 |
| 4 | 71.48 ± 5.44; 71.88 | 71.69 ± 5.71; 72.27 |
| 6 | 68.12 ± 5.31; 69.01 | 68.33 ± 4.88; 69.01 |
| 8 | 63.49 ± 4.07; 63.28 | 63.83 ± 4.12; 62.50 |

The [full comparison](README.md), [summary CSV](summary.csv), and [protocol](protocol.md) retain all 30 configurations and the separate axes, conditional slices and initialisation diagnostics.

## Follow-up: excitation amplitude (2026-09-16)

The gradient discussion motivated a [paired four-event recovery pilot](../amplitude-pilot/README.md):
could learning excitation amplitude release a pitch–time swap? A fresh fixed-amplitude
fit reproduced the swap at 2.048 padding. Learning amplitudes from that state reduced
CeL by 7.25% without resolving the swap; learning them from the original start gave
worse matched errors and one weak excitation. The linked note retains the controls,
full trajectories, checks and limitations. This is a single selected-case diagnostic.
