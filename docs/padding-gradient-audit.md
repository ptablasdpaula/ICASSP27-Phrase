# Figure 3 BiCuL gradient counts versus onset FFT padding

Computed 16 September 2026 using `scripts/audit_padding_directions.py` and
repository synthesis code at `0884dd5` (subsequent `74136ec` edits only paper prose).

The diagnostic imports the figure script's 120 displayed non-target coordinates,
control mapping and target-direction counting function. Target: 160 Hz at
1 second. Both target and candidates use the tested onset FFT length. All other
synthesis and BiCuL settings retain their defaults: Thiran 3/1 propagation,
DF2 recurrence, hard reset, 4000 Hz, 8000 samples, float64.

Execution used the compiled PhilTorch/TorchLPC **CPU** backend, with
PyTorch 2.7.1+cu126. These are fresh CPU calculations, not GPU-confirmed
reruns. A supplementary GPU job remained queued and was cancelled.

| Onset FFT length | Ratio to signal length | Target-directed gradients | Zero gradients | Minimum target dot product |
| ---: | ---: | ---: | ---: | ---: |
| 16,384 | 2.048× | 120/120 | 0 | 0.0434940953 |
| 32,768 | 4.096× | 120/120 | 0 | 0.0469831354 |
| 262,144 | 32.768× | 120/120 | 0 | 0.0481806630 |

The last column is the dot product of unit negative gradient and displacement
towards the target in the figure's normalised pitch/onset coordinates. A
strictly positive value passes the paper's criterion. The original-padding
CPU rerun reproduces the figure's 120/120 count.

The corresponding `padding-directions-*.json` files contain all coordinates,
loss values, gradients, dot products and synthesis configuration. The results
support preserving the displayed count under these padding changes in this
single-target diagnostic; they do not test phrase recovery. Paper figures,
synthesis defaults and campaign results have not been changed.
