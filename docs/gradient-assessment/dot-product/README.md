# Positive-dot-product gradient assessment

[Figure (PNG)](gradient-assessment.png) · [PDF](gradient-assessment.pdf) · [Full statistics](summary.csv)

This rescores the same 66,560 primary candidate–target comparisons using the saved gradients. No synthesis, backward passes, or new sampling were performed. The original componentwise results are preserved in the parent directory.

For each event, success means `(-gradient) dot (matched_target - candidate) > 0`, using coordinates `(log2(f/80), onset_seconds)`. Matching remains unrestricted Hungarian with squared octave error plus squared second error, so 1 s and 1 octave have equal cost. This is a per-event dot product, not a dot product over the whole phrase. Scores average events within candidates, candidates within targets, then targets equally.

A positive dot product allows one component to point away if improvement in the other outweighs it. It indicates a first-order reduction in the squared matched parameter error along the infinitesimal descent direction; it does not establish finite-step improvement or convergence.

| Loss | Both, 1 | Pitch, 2 | Pitch, 4 | Time, 2 | Time, 4 | Both, 2 | Both, 4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| L1 | 54.9 | 59.8 | 54.5 | 50.9 | 51.7 | 52.5 | 51.4 |
| L2 | 49.6 | 57.3 | 52.5 | 51.2 | 53.2 | 52.6 | 52.7 |
| Single STFT | 57.4 | 68.0 | 61.5 | 64.7 | 58.4 | 57.7 | 55.7 |
| Linear MSS | 57.9 | 71.0 | 62.8 | 56.3 | 55.9 | 58.5 | 56.6 |
| Smooth MSS | 58.4 | 78.0 | 67.0 | 54.4 | 54.7 | 58.5 | 56.3 |
| SOT | 50.7 | 74.8 | 66.9 | 50.2 | 50.4 | 50.5 | 50.2 |
| TFW2 | 65.9 | 68.7 | 60.0 | 65.6 | 58.2 | 62.5 | 58.6 |
| log-TFW2 | 68.2 | 71.0 | 62.3 | 59.2 | 55.7 | 61.1 | 57.9 |
| CeL | 98.1 | 84.2 | 72.2 | 80.8 | 68.9 | 82.5 | 72.7 |
| Log-CeL | 98.1 | 85.1 | 73.2 | 81.1 | 69.1 | 82.7 | 73.2 |
| Fade-CeL | 97.9 | 90.7 | 77.8 | 79.9 | 69.8 | 82.3 | 74.0 |

The existing sample counts are retained: 1,024 / 512 / 256 / 256 / 256 / 512 / 256, left to right. They were selected using the earlier componentwise score. The new dot-product repeat checks are descriptive; no new adaptive sampling was performed.

| Column | Maximum independent-seed change (percentage points) |
|---|---:|
| joint, 1 events | 2.441 |
| pitch, 2 events | 0.977 |
| pitch, 4 events | 1.416 |
| time, 2 events | 1.953 |
| time, 4 events | 0.989 |
| joint, 2 events | 1.196 |
| joint, 4 events | 0.830 |

All multi-event columns remain within two percentage points. The centred single-event column reaches 2.441 points. Full per-loss differences are in `sampling.json`. The gradient matching, numerical limitations, raw archives and target distributions are documented in the [parent protocol](../README.md).

Regenerate with:

```bash
PYTHONHOME="$PWD/.pixi/envs/default" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .pixi/envs/default/bin/python scripts/assess_gradients.py report \
  --root results/gradient-assessment-gpu --metric dot_directed \
  --output docs/gradient-assessment/dot-product
```
