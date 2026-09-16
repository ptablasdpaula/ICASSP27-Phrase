# Complete counterfactual takeover

User follow-up 2026-09-16: preserve the refinement that made an alternative
assignment useful by adopting its whole state, not just its pitch permutation.
Selected-case pilot: same C04-T0005 plateau, canonical four-direction CeL,
2.048 padding, independent times, amplitudes fixed at 0.8.

One sweep tests 1↔2, then 2↔3, then 3↔4. Freeze the live fit during each paired
trial. Keep and swap start from clones of the same live parameters, moments,
Adam step count and LR. Run both for the same number of updates: minimum 30,
then stop when BOTH have gone 40 updates without a relative 1e-4 meaningful
improvement, or at a 300-update cap. These settings are pilot choices, not tuned
or established general settings. Trial LR stays .05 initially, no decay during
paired trials. All losses use fixed original plateau-loss conditioning.

For each branch retain its strict-best complete state. Accept the swap only if
its best canonical loss is at least 1% below BOTH the live checkpoint and the
best keep branch. Otherwise adopt keep if it beats live by 1%; else stay put.
Acceptance transfers raw controls, complete permutation, first/second moments,
step count and LR from the SAME selected checkpoint. Exact equality checks
verify isolation during trial and full-state takeover; re-rendering verifies
the accepted loss. This is a hard branch decision, not soft pitch interpolation.

Only after a paired trial finishes do we advance to the next adjacent pair.
After one complete sweep, continue ordinary canonical Adam from the adopted
state for up to 3000 further updates. This continuation preserves moments and
step count initially, then uses plateau patience 100/200, LR factor .3 and
stop patience 250, meaningful relative improvement 1e-4. Later plateau recovery
restores the complete best state, clears moments and reduces LR (floor 1e-5).
This is a research pilot, not an alteration to the registered production fitter.

The original failed state contains controls but no saved live Adam moments;
the first trial therefore starts with zero moments and LR .05. Subsequent
trials and continuation carry the accepted state. Selection never uses target
coordinates or matched errors; only canonical audio loss. Compute is greater
than ordinary descent; no equal-compute performance claim is made.

Run `python scripts/test_counterfactual_takeover.py` with the project Python and
PYTHONHOME set to `.pixi/envs/default`. Results pending.
