# Four-event excitation-amplitude pilot

Requested 2026-09-16. Hypothesis: optimising excitation amplitudes alongside pitch
and onset may release fits whose pitches are associated with the wrong events.
A competing explanation is that amplitudes merely suppress misplaced events.
This is a selected-case diagnostic, not the 150-target recovery comparison.

## Context from the direction screen

The [direction-screen findings](../cel-gradient-screen/findings.md) and its full
CSV preserve the completed 30-variant experiment. The simultaneous-error results
favour downward-frequency directions for pitch: at four events, the downward
pair gives 72.6% pitch / 65.8% onset alignment, versus 69.2% / 64.6% for all four.
In structured pitch-only slices, the downward pair gives 100%, 99.1%, 98.7%
pitch alignment at four/six/eight events, versus 100%, 96.3%, 92.4% for all four.
Log-Weighing raises the latter to 100%, 99.1%, 98.4%. Timing-only results are mixed:
the downward pair reaches 94.7% at four events; forward-time directions with
Log-Weighing reach 92.4% at eight. These local gradient observations do not
establish convergence. No subset has yet been selected for the full rerun.

## Case selection and protocol

Read-only inspection of the old Fourier-onset pilot found C04-T0005 (UI target 6)
with a clear middle-event pitch swap. Target pitches are approximately
239.20, 153.41, 270.20, 125.76 Hz at 0.7839, 1.3096, 1.5059, 1.6056 s.
The old fit gave 238.53, 270.36, 154.75, 125.93 Hz at
0.7839, 1.3096, 1.3892, 1.6065 s. Historical source:
`/gpfs/scratch/acw794/ICASSP2027-PluckedStringPhrase-v2/simple-dwg-v2-unweighted-fourier-onset-pilot-v1/runs/experiment2/task-00095.json`.
It used different padding, so it is only a case-selection source.

Fresh target audio and all new fits use FFT length 16384 / 8000 samples = 2.048,
the current synthesis path, and unweighted four-direction square-root CeL
(the implementation name is `bidirectional_cumulative_energy`). Target
excitation amplitudes remain 0.8. Four fits use the registered Adam/rollback
schedule, each capped at 3000 updates:

1. Fixed amplitudes, registered equal-cell initialisation.
2. Fixed amplitudes, restart at the fresh baseline's strict-best controls.
3. Trainable amplitudes, same restart controls as 2.
4. Trainable amplitudes, original equal-cell initialisation.

Each restart resets Adam moments and learning rate for both conditions.
Trainable amplitude is positive: `a_i = 0.8 exp(z_i)`, initially `z_i = 0`.
There is no gain penalty or upper bound. The same 0.05 initial learning rate
applies to log-amplitude and the registered bounded pitch/onset logits.
Amplitude scales each event's excitation spectrum before summation and the
waveguide; amplitudes follow the same onset-sort permutation. Target-power
normalisation is unchanged. Resets still occur at every candidate onset.

The isolated fitter copies `src/optimization.py` from 9250d9b and leaves the
production experiment unchanged. Verification compares fixed-amplitude audio,
pitch/onset gradients and three Adam updates exactly with production, and
checks amplitude derivatives by central finite differences at unsorted onsets.
JSON trajectories retain all evaluated controls, amplitudes, objective values
and Hungarian-matched pitch/onset MAEs using the paper's assignment cost.
Report strict-best objective states; a lower loss alone is not recovery.

Run with the project Python (set PYTHONHOME to `.pixi/envs/default`):

```
python scripts/test_amplitude_recovery.py --verify
python scripts/test_amplitude_recovery.py --output docs/amplitude-pilot
```

Results are pending. This pilot does not replace the outstanding full recovery
rerun at 2.048 padding.
