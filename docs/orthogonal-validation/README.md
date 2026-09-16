# Switching stalled diagonal CeL fits to axis-only accumulation

Prespecified follow-up to the successful [four-event development case](../orthogonal-escape-pilot/README.md).
Reuse the **same six held-out failed phrases** selected for the
[takeover experiment](../takeover-validation/cases.json): two each with 4, 6,
and 8 events. These are selected failures, not a representative success-rate
sample. Do not replace any case based on its switch outcome.

Use the saved fresh 2.048-padding baseline fits from that experiment. Each
terminated at patience 250 with amplitudes fixed at 0.8. Re-render and verify
its strict-best diagonal loss within 1e-12 before switching. Preserve the
baseline trajectory, renderer provenance and initial metrics in the raw archive.
This implements one switch after a previously established stall, not repeated
automatic switching during fitting.

For each baseline, run two independent restarts:

1. All four axis-only cumulative directions (up, down, right, left).
2. Frequency-only accumulation (up and down, independently in each frame).

Use the identical STFT, global target-power normalisation, square-root features
and directional RMS average as the development pilot. Keep independent bounded
pitch/onset logits and fixed amplitudes. There is no Log-Weighing or pitch swap.
Start fresh Adam at LR .05 and run 1600 exploration updates, dividing by each
restart's initial objective value. Permit uphill diagonal steps, but save the
strict-best original diagonal-loss checkpoint including the starting state.
Then refine that checkpoint using diagonal CeL and the registered Adam schedule
(maximum 3000 updates, plateau patience 100, stop patience 250). Parameter errors
are diagnostic only and never choose checkpoints.

A shared diagonal-only control gets **the larger of the two total switch
budgets**, including exploration and refinement. It repeatedly applies ordinary
canonical descent with the registered patience/LR policy, restarting from the
best controls with fresh moments and LR .05 when patience is exhausted, until
that exact budget is spent. Thus each switch receives no more gradient updates
than its control; the smaller-budget switch may receive fewer. Forward counts
and wall time are not exactly matched. This control is distinct from the
fixed-LR exploration control used in the single-case development pilot.

Primary diagnostic: jointly matched pitch/onset errors and normalized joint
event error relative to the control. Also report canonical loss and relative
waveform L2. Lower loss alone does not establish parameter recovery. Define
strict recovery as pitch MAE <1 cent AND onset MAE <1 ms, and report both
improvements and regressions; do not select the better variant per phrase as
if it were an implementable policy.

Implementation: `scripts/test_orthogonal_directions.py --case-index INDEX VARIANT`
and `scripts/validate_orthogonal_cases.py --index INDEX`. Six indices are 0–5 in
the previously frozen case list. Production fitting code and paper are unchanged.

Results pending.
