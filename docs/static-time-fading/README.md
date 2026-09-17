# Static time-only fading

Complete: seven problem phrases × two fresh methods =14 fits.

**Time horizon stays at 0.25 s from the start. Frequency accumulation is entirely unfaded.**

[Protocol](protocol.md) · [Per-phrase metrics](per_phrase.csv) · [Statistics](summary.csv) · [Full trajectories](raw-results.json.gz)

Four diagonals, fixed amplitudes, no Log-Weighing. Registered Adam and patience from update0, maximum3000 updates. Each method selects its own best fixed training loss.

## Best own training loss

Pitch MAE (cents) / onset MAE (ms).

| Case | Unfaded control | Static time-only .25s |
|---|---:|---:|
| C04-T0005 | 5.455 / 67.378 | 157.687 / 180.092 |
| C04-T0003 | 41.956 / 103.159 | 161.129 / 205.950 |
| C04-T0007 | 153.796 / 6.479 | 0.000 / 0.000 |
| C06-T0000 | 377.992 / 114.196 | 154.387 / 117.153 |
| C06-T0008 | 365.251 / 216.170 | 140.663 / 169.382 |
| C08-T0000 | 99.728 / 116.178 | 62.807 / 62.501 |
| C08-T0004 | 180.933 / 203.400 | 151.501 / 131.220 |

## Best common unfaded loss

Pitch MAE (cents) / onset MAE (ms).

| Case | Unfaded control | Static time-only .25s |
|---|---:|---:|
| C04-T0005 | 5.455 / 67.378 | 121.098 / 181.009 |
| C04-T0003 | 41.956 / 103.159 | 197.279 / 207.215 |
| C04-T0007 | 153.796 / 6.479 | 0.000 / 0.000 |
| C06-T0000 | 377.992 / 114.196 | 181.498 / 120.861 |
| C06-T0008 | 365.251 / 216.170 | 140.749 / 168.918 |
| C08-T0000 | 99.728 / 116.178 | 63.504 / 62.382 |
| C08-T0004 | 180.933 / 203.400 | 119.912 / 140.945 |

## Actual last iterate

Pitch MAE (cents) / onset MAE (ms).

| Case | Unfaded control | Static time-only .25s |
|---|---:|---:|
| C04-T0005 | 8.694 / 71.993 | 149.595 / 173.489 |
| C04-T0003 | 41.970 / 103.148 | 161.860 / 211.920 |
| C04-T0007 | 155.331 / 7.345 | 0.001 / 0.000 |
| C06-T0000 | 377.367 / 114.139 | 154.136 / 117.146 |
| C06-T0008 | 365.283 / 216.170 | 141.446 / 170.282 |
| C08-T0000 | 99.727 / 116.176 | 62.858 / 62.491 |
| C08-T0004 | 225.702 / 208.800 | 150.023 / 171.117 |

Unlike the preceding progressive experiment, these runs enable patience immediately. That earlier experiment used2000 updates before patience. Compare this static method primarily with its newly rerun matched control, not directly with that earlier schedule. No target parameter error selects a checkpoint. One trajectory per setting on selected failures.
