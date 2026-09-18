# `cels-audio`

`cels-audio` provides differentiable Cumulative Energy Losses (CeLs) for
PyTorch. A CeL compares energy accumulated over selected directions of a
frequency--time representation, exposing displacement that pointwise losses
cannot measure when candidate and target energy do not overlap.

## Installation

This pre-release is installed from its tagged Git branch:

```bash
pip install "git+https://github.com/ptablasdpaula/ICASSP27-Phrase.git@cels-v0.1.0"
```

## Waveform use

```python
import torch
from cels import Direction, STFTCumulativeEnergyLoss

loss = STFTCumulativeEnergyLoss(
    sample_rate=16_000,
    n_fft=1024,
    hop_length=256,
    directions=(Direction.RIGHT_UP, Direction.RIGHT_DOWN),
    log_weighting=True,
)

target = torch.randn(32_000)
candidate = torch.randn(32_000, requires_grad=True)
value = loss(candidate, target)
value.backward()
```

For repeated comparisons with one target, cache its transform and cumulative
surfaces:

```python
bound = loss.bind(target)
value = bound(candidate)
```

## Custom representations

`CumulativeEnergyLoss` accepts any nonnegative tensor whose final dimensions
are frequency and time. Supply physical coordinates when using logarithmic
frequency weighting or finite decay horizons.

```python
from cels import ALL_DIRECTIONS, CumulativeEnergyLoss

criterion = CumulativeEnergyLoss(
    directions=ALL_DIRECTIONS,
    log_weighting=True,
    time_decay_seconds=1.0,
    frequency_decay_hz=1000.0,
)
value = criterion(candidate_power, target_power, frequency=f_hz, time=t_seconds)
```

The four diagonal directions accumulate over both axes. `up` and `down`
accumulate frequency independently in every frame; `right` and `left`
accumulate time independently in every frequency bin. A loss averages the
RMS discrepancies of every selected direction.

The helper `paper_loss()` returns the frozen CeL, logCeL, decCeL, and tlogCeL
configurations used in the accompanying ICASSP 2027 experiments.
