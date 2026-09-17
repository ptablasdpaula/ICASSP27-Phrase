# One-second / one-octave fading

Complete: C08-T0000, two starting conditions. Earlier matched controls included below.

[Protocol](protocol.md) · [Qualification](qualification.json) · [Full metrics](summary.csv)

| Start | Method | Pitch MAE (cents) | Onset MAE (ms) | Updates |
|---|---|---:|---:|---:|
| plateau | No fading | 15.798 | 48.300 | 248 |
| plateau | 1 s / 1000 Hz | 15.500 | 49.044 | 458 |
| plateau | 1 s / 1 octave | 12.969 | 55.507 | 1242 |
| fresh | No fading | 99.728 | 116.178 | 1452 |
| fresh | 1 s / 1000 Hz | 0.084 | 0.025 | 3000 |
| fresh | 1 s / 1 octave | 9.737 | 27.794 | 1009 |

All four diagonal directions; logarithmic fading, fixed amplitudes, no Log-Weighing, same registered optimiser/patience and 3000-update cap. Each fit selects the best own-objective checkpoint. One second / one octave denotes zero-weight distance, not half-weight distance.

One phrase only; compare each starting condition separately. See the CSV for common unfaded loss and LSD. [Plateau trajectory](plateau.json.gz) · [Fresh trajectory](fresh.json.gz)
