# Static time-only fading

Seven fresh failed-phrase examples, as in the preceding time-only progressive
pilot. Two methods per phrase: static0.25-second time fade and fully unfaded
four-diagonal control.14 fits, Slurm array27216516. No saved plateau initialises
any run. All event pitches start160Hz with registered equal-cell onsets.

The time kernel is w(d;H)=max(0,1-log(1+9*d/H)/log(10)), H=.25s throughout.
No fade schedule or warm-up. The horizon is the distance at which weight reaches
zero, not its half-weight distance. Frequency accumulation has unit weights
for all bins in each directional rectangle, with no decay or finite horizon.
Original global target power normalisation, square-root features, equal mean
of four directional RMSs. No Log-Weighing, fixed amplitudes .8,2.048 onset
padding, independent bounded pitch/onset logits, CPU float64 single-thread
compiled TorchLPC and deterministic algorithms.

Both methods use the unchanged registered fitter: Adam LR .05, meaningful
relative improvement1e-4, rollback/clear moments/LR x.3 at patience100/200,
stop250 or3000 updates. Patience is enabled from the start. Condition each
objective by its own initial value. The primary selected state minimises
its own FIXED training objective over the trajectory.

Also track each run's strict-best original UNFADED diagonal loss and report
that checkpoint independently. This diagnostic never affects optimisation.
Finally report the actual last iterate. No target parameter error selects a
checkpoint. Selected own-loss and common-loss checkpoints pass rerendering.

Qualification requires the unchanged previously checked time-only objective
and gradients, and current registered optimiser/LSD qualification. Explicitly
verify unit frequency weights and zero self-loss/self-gradient. The prior
qualification already checked exact terminal0.25s time-only value/gradients
and audio finite differences. Reuse that exact implementation without edits.

The controls are rerun with matching initialisation, optimiser and budget.
Do not confuse these with the preceding progressive pilot's controls, which
forced2000 updates before enabling patience. Differences versus that earlier
pilot combine static/adaptive fading with different stopping/LR schedules.

Report per-phrase pitch/onset MAE, joint RMS, LSD, update counts, all three
checkpoint views, and mean/sampleSD/median. Report all seven and the other five
excluding the original C04-T0005/C08-T0000 development/exploration cases.
One trajectory per setting on selected failures; no generalisation claim.
