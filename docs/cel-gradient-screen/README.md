# CeL gradient screening at 2.048× onset padding

All 30 direction subsets/Log-Weighing configurations were tested without optimisation. The square-root feature and all other synthesis/loss settings are fixed.

13099 candidate phrases; 0 tied Hungarian assignments excluded; 0 nonfinite variant/candidate gradients.

Start with the [findings](findings.md) and [execution/validation notes](execution.md). Plots: [simultaneous errors](independent-simultaneous.png), [pitch-only slices](structured-isolated_pitch.png), [timing-only slices](structured-isolated_timing.png), [joint slices](structured-isolated_joint.png), and [fit initialisation](independent-initialisation.png).

Direction bits: 1 = right/up, 2 = right/down, 4 = left/up, 8 = left/down. Add bits to identify a subset; `_lw` adds Log-Weighing. `cel_15` is the existing four-direction variant.

## Independent targets, simultaneous perturbations

Entries are mean target-directed event percentages, separately averaged within each phrase then across 30 independent targets at each event count. Each cell is pitch / onset / joint. The complete CSV also includes sample standard deviations, medians, cosine alignment, whole-phrase alignment, zero gradients and drift of already-correct coordinates.

| Directions | 1 event | 2 events | 4 events | 6 events | 8 events |
|---|---:|---:|---:|---:|---:|
| ↗ | 82.9 / 94.3 / 99.0 | 72.1 / 76.1 / 82.4 | 65.8 / 64.9 / 70.3 | 64.5 / 60.7 / 66.4 | 59.9 / 58.7 / 62.6 |
| ↘ | 84.8 / 93.2 / 96.2 | 75.2 / 75.6 / 81.7 | 69.9 / 66.1 / 71.3 | 67.8 / 60.8 / 66.1 | 64.2 / 58.8 / 62.9 |
| ↗ ↘ | 83.4 / 94.5 / 99.2 | 72.8 / 76.1 / 82.3 | 66.4 / 65.4 / 70.7 | 65.2 / 61.1 / 66.9 | 60.6 / 58.9 / 63.2 |
| ↖ | 83.6 / 73.3 / 97.0 | 72.9 / 63.1 / 75.6 | 68.1 / 55.3 / 64.9 | 67.6 / 53.7 / 62.4 | 62.2 / 52.9 / 59.6 |
| ↗ ↖ | 84.9 / 91.0 / 98.4 | 74.3 / 76.4 / 83.4 | 68.4 / 64.3 / 71.1 | 68.0 / 59.9 / 67.5 | 62.5 / 58.3 / 63.3 |
| ↘ ↖ | 85.8 / 80.7 / 97.8 | 74.3 / 71.4 / 81.5 | 69.5 / 62.0 / 69.9 | 68.9 / 58.1 / 66.2 | 63.4 / 57.4 / 62.8 |
| ↗ ↘ ↖ | 83.6 / 93.6 / 98.5 | 74.3 / 76.8 / 83.9 | 69.0 / 64.9 / 71.4 | 68.4 / 60.5 / 67.8 | 63.0 / 58.6 / 63.8 |
| ↙ | 83.3 / 82.9 / 96.9 | 75.1 / 65.9 / 76.9 | 71.0 / 57.3 / 66.0 | 69.7 / 55.1 / 62.6 | 64.5 / 53.2 / 58.9 |
| ↗ ↙ | 84.3 / 94.3 / 99.2 | 74.5 / 77.1 / 84.0 | 68.8 / 65.8 / 72.1 | 67.8 / 60.5 / 67.3 | 62.3 / 58.1 / 62.6 |
| ↘ ↙ | 86.0 / 94.4 / 98.1 | 76.7 / 76.7 / 83.1 | 72.6 / 65.8 / 72.4 | 70.9 / 60.5 / 67.4 | 66.4 / 57.6 / 62.6 |
| ↗ ↘ ↙ | 84.4 / 95.3 / 99.2 | 75.1 / 77.0 / 83.9 | 69.3 / 66.5 / 72.3 | 68.1 / 61.2 / 67.9 | 62.9 / 58.5 / 63.3 |
| ↖ ↙ | 83.8 / 75.6 / 97.2 | 73.1 / 63.9 / 76.0 | 68.4 / 56.2 / 65.5 | 68.1 / 54.1 / 63.1 | 62.6 / 53.1 / 59.9 |
| ↗ ↖ ↙ | 85.9 / 90.1 / 98.2 | 74.7 / 76.3 / 83.3 | 69.2 / 64.2 / 71.1 | 68.6 / 59.4 / 67.5 | 62.8 / 57.6 / 62.8 |
| ↘ ↖ ↙ | 85.7 / 80.9 / 97.8 | 74.6 / 71.5 / 81.0 | 69.7 / 62.0 / 69.9 | 69.0 / 57.8 / 66.4 | 63.6 / 56.8 / 62.5 |
| ↗ ↘ ↖ ↙ | 85.2 / 92.0 / 98.5 | 74.8 / 77.0 / 83.9 | 69.2 / 64.6 / 71.5 | 68.8 / 60.2 / 68.1 | 63.3 / 58.3 / 63.5 |
| ↗ + LW | 83.8 / 93.1 / 99.1 | 73.6 / 74.8 / 81.8 | 66.5 / 63.3 / 69.6 | 66.3 / 59.8 / 66.5 | 61.3 / 57.3 / 62.0 |
| ↘ + LW | 83.6 / 94.3 / 97.8 | 73.1 / 76.3 / 81.6 | 68.0 / 64.9 / 70.4 | 67.1 / 61.4 / 67.2 | 62.9 / 58.8 / 62.6 |
| ↗ ↘ + LW | 83.6 / 94.0 / 98.8 | 73.4 / 76.3 / 82.3 | 67.4 / 65.1 / 70.8 | 66.8 / 61.0 / 67.4 | 61.7 / 58.8 / 63.2 |
| ↖ + LW | 85.3 / 72.6 / 97.8 | 74.2 / 61.5 / 75.4 | 69.5 / 55.5 / 65.9 | 68.5 / 54.4 / 63.6 | 63.1 / 53.9 / 60.6 |
| ↗ ↖ + LW | 85.8 / 91.0 / 98.5 | 76.0 / 74.6 / 83.2 | 69.8 / 62.6 / 70.6 | 70.1 / 59.8 / 68.0 | 63.5 / 58.0 / 63.6 |
| ↘ ↖ + LW | 85.8 / 89.9 / 98.3 | 75.9 / 75.5 / 83.2 | 70.0 / 63.4 / 71.1 | 70.3 / 60.1 / 68.1 | 64.5 / 58.3 / 64.0 |
| ↗ ↘ ↖ + LW | 83.4 / 94.1 / 99.2 | 75.4 / 76.3 / 84.0 | 69.4 / 64.3 / 72.0 | 70.1 / 60.5 / 68.5 | 63.8 / 58.5 / 64.2 |
| ↙ + LW | 83.6 / 77.2 / 96.6 | 73.9 / 63.9 / 75.7 | 70.4 / 55.1 / 64.5 | 69.7 / 53.7 / 61.7 | 64.6 / 51.9 / 57.8 |
| ↗ ↙ + LW | 84.6 / 92.5 / 98.1 | 75.8 / 76.7 / 83.0 | 70.2 / 63.2 / 70.8 | 69.9 / 59.4 / 67.2 | 63.6 / 56.1 / 61.9 |
| ↘ ↙ + LW | 84.8 / 92.2 / 98.0 | 75.4 / 76.5 / 83.1 | 71.3 / 64.6 / 71.0 | 70.4 / 60.5 / 67.5 | 65.3 / 57.3 / 62.2 |
| ↗ ↘ ↙ + LW | 83.4 / 94.8 / 98.8 | 74.9 / 77.3 / 84.3 | 69.8 / 64.6 / 71.9 | 69.9 / 61.1 / 68.3 | 64.2 / 58.1 / 63.3 |
| ↖ ↙ + LW | 84.7 / 74.4 / 97.1 | 74.4 / 64.0 / 76.5 | 69.7 / 55.4 / 65.8 | 69.5 / 54.1 / 63.0 | 63.8 / 53.0 / 60.1 |
| ↗ ↖ ↙ + LW | 88.0 / 86.7 / 98.0 | 75.8 / 74.6 / 82.7 | 70.4 / 62.8 / 70.2 | 70.5 / 59.2 / 67.3 | 64.2 / 56.8 / 62.8 |
| ↘ ↖ ↙ + LW | 86.8 / 86.5 / 98.0 | 75.8 / 74.5 / 82.8 | 70.5 / 63.4 / 70.6 | 70.5 / 60.0 / 67.8 | 65.0 / 57.7 / 63.1 |
| ↗ ↘ ↖ ↙ + LW | 85.1 / 91.7 / 98.4 | 76.0 / 76.8 / 83.8 | 70.2 / 64.2 / 71.7 | 70.8 / 60.4 / 68.3 | 64.3 / 58.0 / 63.8 |

## Interpretation limits

These are local derivatives in normalised physical coordinates, not bounded logits or Adam updates. The implemented onset-state resets remain detached: gradients are conditional on the current discrete reset/order configuration. Positive alignment does not establish convergence or finite-step improvement.

Single-event slices hold other events correct; the drift statistic measures gradients that move already-correct coordinates. Whole-phrase alignment can be positive even when some events or axes move away.

Correct-coordinate masks are evaluated after Hungarian matching. A timing-only perturbation can change the assignment in the eight-event ascending/descending cases, introducing matched pitch displacement. This accounts for the eligible pitch scores in those timing-only slices.

Pilot targets use seed 2028 and are independent of the frozen 150-target evaluation registry. No optimisation runs or final variant selection have been made. Existing phrase-recovery results remain from the earlier padding setting.

Raw elementary losses/gradients, target/candidate coordinates, assignments and backend provenance are in `raw-results.zip`; `provenance.json` records hashes and subset coefficients. Reconstruct each variant by averaging its elementary terms. Blank summary cells denote no eligible coordinates or insufficient phrases for sample standard deviation.
