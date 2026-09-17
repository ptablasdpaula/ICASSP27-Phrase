# Comparison with the original unfaded diagonal results

Reference: the saved original diagonal fit for each phrase, evaluated with the same current metrics. Negative changes in the CSV mean improvement. This reference differs from the 1000-Hz comparator used in paired.csv.

| Start | Fade | Lower joint error | Higher joint error | Lower pitch AND onset MAE |
|---|---|---:|---:|---:|
| plateau | 1000hz | 4/7 | 3/7 | 3/7 |
| plateau | 2oct | 5/7 | 2/7 | 4/7 |
| plateau | 4oct | 4/7 | 3/7 | 2/7 |
| fresh | 1000hz | 5/7 | 2/7 | 5/7 |
| fresh | 2oct | 3/7 | 4/7 | 2/7 |
| fresh | 4oct | 4/7 | 3/7 | 5/7 |

The plateau comparison measures what happened after changing the objective at the original solution; it includes additional optimisation. Fresh fits are compared with historical saved diagonal fits, not a newly rerun matched control for every phrase. Earlier numerical-sensitivity limitations therefore apply.

Joint error is RMS over matched normalised pitch/onset distances. It can worsen even when both separate MAEs improve, because RMS and MAE aggregate event errors differently.

[Per-case changes](versus-original-diagonal.csv)
