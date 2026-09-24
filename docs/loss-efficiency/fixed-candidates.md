# Fixed-candidate efficiency benchmark (2026-09-24)

Supersedes the Adam benchmark for loss-cost comparisons. All five tasks of GPU array 28148618 completed successfully on NVIDIA A100-PCIE-40GB. Each loss uses 150 frozen one-note targets in 15 batches of 10; every candidate stays at 160 Hz and 1 s. Five warm-up passes precede 20 measured render–loss–backward passes. No optimiser, parameter updates or accumulated gradients. Target precomputation is excluded. All batches passed unchanged-parameter and finite-gradient checks.

Medians across batches; percentages are increases over SS.

| Loss | Time (ms) | Increase | Peak allocated memory (GiB) | Increase |
|---|---:|---:|---:|---:|
| SS | 374.13 | +0.00% | 1.805 | +0.00% |
| MSS | 379.77 | +1.51% | 1.830 | +1.35% |
| SOT | 379.91 | +1.54% | 1.839 | +1.88% |
| TFW2 | 374.36 | +0.06% | 1.808 | +0.14% |
| CeL | 374.31 | +0.05% | 1.824 | +1.03% |

Raw data: `results/loss-efficiency/fixed-28148618_{0,1,2,3,4}.json`. The paper efficiency table has not yet been replaced.
