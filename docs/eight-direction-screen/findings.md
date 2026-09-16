# Descriptive findings

Values below average the target-level percentages equally over **2, 4, 6 and 8 events**. They summarise multi-event trends, not recovery success. The full CSV retains each cardinality, target-level sample SD and median. Pitch-only and onset-only conditions perturb one event while keeping other events correct; simultaneous errors perturb all events.

- Orthogonal directions help conditionally, but orthogonal-only losses do not dominate the diagonal loss. With Log-Weighing, ↑ reaches 100% pitch alignment in pitch-only slices, versus 99.38% for four diagonals. With simultaneous errors, the same ↑ loss gives only 62.05% pitch alignment versus 70.31% for the diagonals.
- Adding all four orthogonals improves simultaneous pitch alignment: 69.05% to 71.05% without Log-Weighing, and 70.31% to 72.66% with it. Joint alignment does not improve (71.75% to 71.50%, and 71.90% to 70.80%, respectively).
- The exploratory mixed subset ↗ ↘ ↑ ↓ + LW is interesting for timing and joint slices: versus four diagonals + LW, onset-only alignment rises from 90.43% to 93.80%, isolated joint alignment from 89.76% to 92.07%, and simultaneous joint alignment from 71.90% to 72.65%. Simultaneous pitch alignment falls from 70.31% to 68.42%. It is a trade-off, not a uniformly better loss.
- These findings support considering different objectives under different error conditions. They do not establish that an automatic switch can identify those conditions or improve recovery; the failed-phrase pilots address a different question.

## Fixed comparison groups

| Directions | Pitch-only: pitch | Onset-only: onset | Joint slice: joint | Simultaneous: pitch | Simultaneous: onset | Simultaneous: joint |
|---|---:|---:|---:|---:|---:|---:|
| ↗ ↘ ↖ ↙ | 97.18% | 89.66% | 86.84% | 69.05% | 65.04% | 71.75% |
| ↑ | 98.05% | 91.27% | 88.02% | 61.41% | 65.15% | 67.25% |
| ↓ | 99.52% | 81.91% | 81.11% | 59.60% | 62.40% | 63.66% |
| ↑ ↓ | 98.92% | 89.20% | 87.39% | 61.44% | 65.42% | 67.00% |
| → | 93.16% | 78.67% | 76.72% | 63.23% | 55.11% | 57.86% |
| ← | 93.11% | 76.79% | 75.73% | 63.34% | 50.80% | 55.14% |
| → ← | 93.29% | 78.33% | 77.01% | 64.49% | 53.56% | 57.29% |
| ↑ ↓ → ← | 99.01% | 85.92% | 83.66% | 66.40% | 59.39% | 61.78% |
| ↗ ↘ ↖ ↙ ↑ ↓ → ← | 97.97% | 89.55% | 89.18% | 71.05% | 65.06% | 71.50% |
| ↗ ↑ | 97.10% | 91.23% | 88.95% | 66.65% | 67.24% | 72.11% |
| ↗ ↘ ↑ ↓ | 97.53% | 91.96% | 89.88% | 67.32% | 67.88% | 72.50% |
| ↗ ↘ ↖ ↙ + LW | 99.38% | 90.43% | 89.76% | 70.31% | 64.84% | 71.90% |
| ↑ + LW | 100.00% | 89.80% | 87.98% | 62.05% | 63.88% | 66.68% |
| ↓ + LW | 99.46% | 86.99% | 85.38% | 60.80% | 64.41% | 66.13% |
| ↑ ↓ + LW | 99.92% | 89.87% | 87.24% | 61.96% | 65.71% | 67.39% |
| → + LW | 94.64% | 82.33% | 77.44% | 65.07% | 55.63% | 59.32% |
| ← + LW | 94.82% | 82.19% | 76.16% | 64.80% | 50.95% | 56.34% |
| → ← + LW | 95.01% | 82.57% | 77.87% | 66.04% | 53.58% | 58.77% |
| ↑ ↓ → ← + LW | 98.81% | 86.09% | 82.15% | 67.72% | 57.73% | 62.52% |
| ↗ ↘ ↖ ↙ ↑ ↓ → ← + LW | 99.62% | 90.55% | 90.37% | 72.66% | 63.65% | 70.80% |
| ↗ ↑ + LW | 99.46% | 92.55% | 91.62% | 68.20% | 66.09% | 71.63% |
| ↗ ↘ ↑ ↓ + LW | 99.54% | 93.80% | 92.07% | 68.42% | 67.45% | 72.65% |

## Exploratory leaders

Top five configurations per measure, ranked on these same candidates. Differences are percentage points relative to the uniform four-diagonal loss. Searching 510 configurations makes these descriptive leaders, not independently validated choices. Several variants can tie. Other-axis drift and component alignment should be checked before interpreting a high joint score.

### structured / isolated_pitch / pitch_directed

| Subset | Target-directed | Change |
|---|---:|---:|
| ↑ + LW | 100.00% | +2.82 pp |
| ↘ ↑ → + LW | 100.00% | +2.82 pp |
| ↑ ↓ → + LW | 100.00% | +2.82 pp |
| ↘ ↑ ↓ → + LW | 100.00% | +2.82 pp |
| ↘ ↙ ↑ → ← + LW | 100.00% | +2.82 pp |

### structured / isolated_timing / onset_directed

| Subset | Target-directed | Change |
|---|---:|---:|
| ↘ ↑ ↓ + LW | 93.94% | +4.28 pp |
| ↘ ↑ | 93.88% | +4.22 pp |
| ↗ ↘ ↙ ↑ ↓ → + LW | 93.81% | +4.14 pp |
| ↗ ↘ ↑ ↓ + LW | 93.80% | +4.14 pp |
| ↘ ↑ + LW | 93.73% | +4.06 pp |

### structured / isolated_joint / joint_directed

| Subset | Target-directed | Change |
|---|---:|---:|
| ↗ ↘ ↑ ↓ + LW | 92.07% | +5.24 pp |
| ↗ ↑ ↓ + LW | 91.68% | +4.85 pp |
| ↗ ↘ ↓ + LW | 91.62% | +4.79 pp |
| ↗ ↑ + LW | 91.62% | +4.79 pp |
| ↗ ↘ ↑ + LW | 91.53% | +4.70 pp |

### independent / simultaneous / pitch_directed

| Subset | Target-directed | Change |
|---|---:|---:|
| ↘ ↖ ↙ ↑ → ← + LW | 72.95% | +3.90 pp |
| ↗ ↖ ↙ → ← + LW | 72.91% | +3.86 pp |
| ↘ ↖ ↙ ↑ ↓ → ← + LW | 72.90% | +3.86 pp |
| ↘ ↖ ↙ → ← + LW | 72.90% | +3.85 pp |
| ↗ ↖ ↙ ↑ → ← + LW | 72.87% | +3.82 pp |

### independent / simultaneous / onset_directed

| Subset | Target-directed | Change |
|---|---:|---:|
| ↗ ↘ ↑ ↓ | 67.88% | +2.84 pp |
| ↘ ↑ ↓ | 67.81% | +2.77 pp |
| ↘ ↑ ↓ + LW | 67.73% | +2.69 pp |
| ↘ ↑ | 67.68% | +2.65 pp |
| ↘ ↓ + LW | 67.62% | +2.59 pp |

### independent / simultaneous / joint_directed

| Subset | Target-directed | Change |
|---|---:|---:|
| ↗ ↘ ↑ ↓ + LW | 72.65% | +0.90 pp |
| ↗ ↘ ↖ ↑ ↓ + LW | 72.52% | +0.77 pp |
| ↗ ↘ ↙ ↑ ↓ | 72.52% | +0.77 pp |
| ↗ ↘ ↑ ↓ | 72.50% | +0.75 pp |
| ↘ ↖ ↑ ↓ + LW | 72.47% | +0.72 pp |

## Reading the results

All direction subsets average raw directional RMS losses equally. A mixture does not force equal gradient strength from the two families; the axis-only surfaces often have smaller scale. Log-Weighing changes spatial quadrature rather than mass accumulation. No frame-wise or frequency-wise normalisation is introduced.

The original assignment rule is preserved. In some eight-event timing-only slices, reassignment creates matched pitch displacement despite unchanged input pitch coordinates. A target-directed joint gradient is not evidence that every component points toward its target, nor that optimisation will escape a plateau.
