# Findings on C08-T0000

Logarithmic fading recovered this phrase almost exactly in two of the four
fresh-start settings. The 1 s / 1000 Hz horizon reached 0.084 cents / 0.025 ms;
the .25 s / 250 Hz horizon reached 0.110 cents / 0.040 ms. The matched ordinary
four-diagonal control reached 99.728 cents / 116.178 ms. All four fresh-start
fade settings improved both pitch and onset MAE versus that control, but the
2 s and .5 s settings retained errors (22.771 / 18.075 and 10.650 / 18.899).
These are fade-to-zero horizons, not half-weight distances: half-weight is
approximately 0.2403 times each horizon.

Using fading after the saved plateau was less successful. The .5 s / 500 Hz
setting reduced onset MAE from 48.300 to 16.753 ms and joint RMS error by about
67.7%, but pitch MAE remained approximately unchanged (15.798 to 16.177 cents).
The shortest horizon increased pitch MAE to 42.077 cents. None of the four
plateau restarts recovered all events accurately. The matched ordinary
continuation retained the starting incumbent.

The common unfaded diagonal loss also falls strongly for the two successful
fresh-start runs: 2.808e-5 and 5.527e-5, versus .010027 for the fresh control.
LSD falls from 17.361 dB to .967 and .911 dB, respectively. Thus the success is
visible in common metrics, not merely smaller values of a different objective.
The .25 s setting stopped on patience after 1344 updates, versus 1452 for the
ordinary fresh control; the 1 s setting used the full 3000-update cap.

Interpretation: this is evidence that localising the accumulation can help
avoid the wrong solution on this phrase. It is weaker evidence for using it
as an escape mechanism once stuck. The dependence on scale is not monotonic;
choosing the best of four settings on one target does not establish a generally
better loss. The two starts answer distinct questions and should not be pooled.

The fresh control differs from the historical run that supplied the saved
plateau. The current control uses the current four-direction objective route
and CPU execution; prior work already found strong trajectory sensitivity to
round-off-scale arithmetic differences. This pilot does not isolate that
historical difference. Compare current fresh runs with their current fresh
control, and plateau runs with their exact saved starting coordinates. This
is one run per setting, without cross-machine or repeat-run robustness checks.

All ten jobs completed successfully (Slurm 27188956, exit 0). Qualification
passed directional weighted-sum, no-fade-limit, self-loss and gradient tests;
all saved best losses were checked by rerendering. The result archive includes
all iterates, settings and selected parameters. No paper or production-loss
source was changed.
