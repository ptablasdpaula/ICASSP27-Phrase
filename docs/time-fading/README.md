# Time-only fading from a fresh start

Complete: seven phrases × two methods = 14 fresh fits.

[Protocol](protocol.md) · [Per-phrase metrics](per_phrase.csv) · [Statistics](summary.csv) · [Full trajectories](raw-results.json.gz)

Only the time horizon shrinks from infinity to 0.25 seconds over 2000 updates. Both methods share the schedule and enable patience after update 2000. Frequency accumulation remains completely unfaded. Fresh starts, fixed amplitudes, no Log-Weighing.

## Best common-loss checkpoint

Both methods select the strict-best original unfaded diagonal loss across the entire run. Entries are pitch MAE (cents) / onset MAE (ms).

| Case | Unfaded control | Time-only fading |
|---|---:|---:|
| C04-T0005 | 6.227 / 70.012 | 0.187 / 0.036 |
| C04-T0003 | 43.191 / 101.795 | 0.398 / 0.104 |
| C04-T0007 | 12.133 / 2.683 | 490.513 / 179.884 |
| C06-T0000 | 7.308 / 45.614 | 46.093 / 36.291 |
| C06-T0008 | 304.312 / 215.278 | 49.864 / 180.308 |
| C08-T0000 | 86.430 / 43.842 | 67.116 / 18.690 |
| C08-T0004 | 180.933 / 203.400 | 203.423 / 299.703 |

## Actual last iterate

No best-checkpoint selection in this table.

| Case | Unfaded control | Time-only fading |
|---|---:|---:|
| C04-T0005 | 6.447 / 72.937 | 0.161 / 0.080 |
| C04-T0003 | 52.320 / 100.761 | 0.794 / 0.287 |
| C04-T0007 | 7.539 / 3.684 | 454.234 / 254.656 |
| C06-T0000 | 11.337 / 45.785 | 91.651 / 30.348 |
| C06-T0008 | 421.506 / 259.516 | 261.448 / 116.140 |
| C08-T0000 | 90.515 / 53.444 | 107.736 / 22.271 |
| C08-T0004 | 358.399 / 345.376 | 276.011 / 221.999 |

![Horizon schedule](schedule.png)

![Best common loss](trajectories.png)

One trajectory per setting on selected failures. The common-loss checkpoint may precede the final time horizon. Final patience monitors each fixed terminal objective, while the reported checkpoint uses the same unfaded metric for both methods. The controls use this experiment’s schedule, not the earlier registered early-stopping schedule.
