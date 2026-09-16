# Axis-only cumulative losses after a diagonal plateau

Requested 2026-09-16. Test C04-T0005 from the strict-best stalled controls in
[the amplitude pilot](../amplitude-pilot/baseline.json). No pitch swaps,
amplitude learning, onset ordering, or Log-Weighing. Padding remains 16384/8000
= 2.048. This is a selected-case escape diagnostic, not a population benchmark.

For STFT power P[k,n], the four new surfaces are:

- Up: sum over i <= k of P[i,n], independently in each frame n.
- Down: sum over i >= k of P[i,n], independently in each frame n.
- Right: sum over j <= n of P[k,j], independently in each frequency bin k.
- Left: sum over j >= n of P[k,j], independently in each frequency bin k.

Use the same STFT as the diagonal CeL (256-sample Hann, hop 64, center=False),
total target STFT power m, feature sqrt(max(S/m, 1e-12)), and RMS feature error.
Each surface uses the same global m, with no per-frame or per-bin mass
normalisation. Average either all four axis-only RMSs or only up/down. The
control averages the original four diagonal RMSs. Time-only accumulation is
along the entire row, not just the immediately adjacent frame. Frequency-only
accumulation leaves frames separate, but the synthesiser still couples pitch
and onset gradients; this does not constrain updates to pitch alone.

All three restart from identical stalled controls with fresh Adam, LR .05,
betas (.9,.999), epsilon 1e-8. Divide each objective by its own initial value.
Run 1600 updates without rollback or LR reduction. Record every iterate and
allow increases in the diagonal loss. Keep the strict-best **original diagonal
CeL** checkpoint (including the initial state), then refine it with the original
CeL and registered Adam schedule (maximum 3000 updates, plateau patience 100,
stop patience 250). No ground-truth parameter errors enter checkpoint selection.
Exploration budgets match; refinement lengths may differ and are reported.

The switch is triggered manually at the previously established plateau; this
experiment does not implement a general automatic stall detector or recurring
rotation schedule. It tests replacing the diagonals by the specified axis-only
objective for one exploration phase, then returning to diagonals.

Validation checks inclusive axis sums against an asymmetric toy grid, zero
self-distance, finite optimisation gradients, and reproduction of the archived
starting diagonal loss to 1e-12. Production objectives and paper are unchanged.

Run `python scripts/test_orthogonal_directions.py VARIANT`, with VARIANT one of
`all_four`, `frequency_only`, `diagonal_control`, in the project environment.
Run `python scripts/report_orthogonal_directions.py` to reproduce the report.
