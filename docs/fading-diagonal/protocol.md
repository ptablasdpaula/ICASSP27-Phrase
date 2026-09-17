# Fading-diagonal pilot protocol

Target: C08-T0000 (load_target(8, 1)), the frozen eight-event failure previously
used in the clockwise pilot. Two initial states: the exact saved best diagonal
plateau from results/takeover-validation/raw/C08-T0000/baseline.json, and the
registered 160-Hz/equal-cell initial guess. Five predeclared objectives for each
start, ten fits total. Slurm array 27188956.

## Accumulation

For physical displacement d >= 0, use

    w(d; H) = max(0, 1 - log(1 + 9 d/H) / log(10)).

The contribution of a power bin to a later bin in a chosen directional rectangle
is multiplied by w(time displacement; H_t) * w(frequency displacement; H_f).
The current bin retains unit weight. Weight reaches zero at horizon H, and
half-weight at H * (sqrt(10)-1)/9, approximately 0.2403 H. This is one explicit
interpretation of logarithmic fading, not an exponential leaky integrator.
Distances use seconds and linear Hz, not log-frequency coordinates. Fading
applies to target and candidate identically; no energy crosses a directional
boundary. Inclusive separable causal matrix multiplication avoids FFT wrap-around.

Settings (H_t seconds, H_f Hz): (2,2000), (1,1000), (.5,500), (.25,250).
The no-fade control uses the exact original cumulative-sum implementation.
All four diagonals are equally averaged, with the usual square-root feature and
1e-12 floor. Normalise by the original total target STFT power, never by faded
mass or a separate per-frame mass. STFT: periodic Hann, FFT256, hop64, center=False.
No Log-Weighing: the only objective change in this pilot is fading accumulation.
The resulting surfaces are weighted local sums, not ordinary cumulative mass/CDFs.

## Optimisation and selection

All runs use fixed .8 amplitudes and 2.048 onset padding (FFT16384), CPU float64,
compiled TorchLPC, deterministic algorithms, single thread. Same qualified
registered fitter, Adam LR .05, initial-own-loss scaling, meaningful relative
improvement 1e-4, rollback/clear moments/LR x.3 at patience100/200, stop250 or3000
updates. Fresh optimiser for each run, including the plateau control. No refinement.
Select strict best OWN training loss, including its initial state, and evaluate
matched pitch/onset errors only afterwards. Save the original unfaded diagonal
loss and the original LSD as common evaluation metrics. Do not compare raw
training-loss magnitudes between different kernels. Joint error uses the paper's
RMS matched error, not mean matched distance from the older escape pilot.

This isolates changing the objective at a plateau and using it from initialisation.
The older clockwise pilot used another continuation schedule, so its final control
is historical context, not the matched control here. One target and one trajectory
per setting: exploratory evidence only. The four scales are a small joint sweep;
frequency-only/time-only fading and broader validation are not tested here.

## Verification and provenance

Explicit nested weighted sums on a small grid agree with all four matrix scans;
no-fade matrices reproduce directional cumulative sums; autograd gradcheck and an
audio finite difference agree; identical audio has exactly zero loss and gradient.
The run checks the saved best loss by rerendering. Source signature includes the
new runner, the unchanged shared-study sources/registry, and the saved plateau.
Existing full-study sources and experiments remain unchanged.
