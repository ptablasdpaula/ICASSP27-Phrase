# Findings: fixed 0.25-second time-only fading

All 14 fresh fits completed successfully. Time fading is fixed at 0.25 s from
the first update, with frequency accumulation entirely unfaded. Both methods
use the registered immediate-patience schedule and their own best training loss.

Static fading improves joint RMS error on 5/7 examples and worsens it on 2/7.
The mean is essentially unchanged (.164621 unfaded versus .164897 static),
while median worsens (.131327 versus .194820). The two regressions are large,
so the majority of per-case gains does not imply an aggregate improvement.

C04-T0007 is recovered to below .001 cents and .001 ms, and remains recovered
at the last iterate and the best common-unfaded-loss checkpoint. The unfaded
control remains at 153.796 cents / 6.479 ms. This is the only phrase meeting
the combined <1-cent/<1-ms criterion; the control recovers none.

The static setting fails to recover the two phrases that the previous gradual
time-only schedule solved: C04-T0005 is 157.687 cents / 180.092 ms (control
5.455/67.378), and C04-T0003 is 161.129/205.950 (control41.956/103.159).
Other static results are C06-T0000 154.387/117.153, C06-T0008 140.663/169.382,
C08-T0000 62.807/62.501, and C08-T0004 151.501/131.220. Full paired controls
and metrics are in README.md and per_phrase.csv.

Using the common unfaded loss to select checkpoints instead of each own loss
still gives 5/7 gains, 2/7 regressions, and the same single recovery. Thus the
main qualitative conclusion is not just a checkpoint-metric artefact. Actual
last iterates give 4/7 gains and 3/7 regressions, with the same recovery.

The fixed and progressive methods solve different cases, but this is NOT a
pure static-versus-scheduled ablation: the progressive experiment disabled
patience and LR reductions until update2000, whereas this static experiment
uses the ordinary settings immediately. Each has matched controls under its
own schedule. Do not compare their control rows as though they were identical.

No frequency fade or Log-Weighing was introduced; amplitudes remain fixed.
Source/coverage validation, matched initial-coordinate checks, and rerendering
of both selected checkpoints passed. Shared objective/gradient and optimiser
qualification remained current. All seven cases are selected failures, with
one trajectory per setting, so this does not establish broader reliability.
