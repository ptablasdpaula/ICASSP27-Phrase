# Submitted numerical results

These data were produced before repository restructuring. Original provenance
records retain their original source hashes and paths as historical evidence;
those paths are not runtime dependencies. `manifest.json` validates the archived
files independently of the current code signature. Newly computed campaigns
record the current source signature and write outside this directory.

| Paper artifact | Numerical inputs | Reproduction command |
|---|---|---|
| Fig. 1 cumulative surfaces | Deterministic 20–1000-Hz sweep; `cumulative-surfaces.json` | `pixi run figures` |
| Fig. 2 waveguide | Static `paper/figures/Waveguide.pdf` | Retained original asset |
| Fig. 3 loss slices | `loss-slices/loss_sweeps.npz` | `pixi run slices`, then `figures --data results` |
| Fig. 4 gradient cosine | `gradient-analysis/phrase-cosine.csv`, `gradient-limitations/summary.csv` | `pixi run gradients all`, then `figures --data results` |
| Table I recovery | `recovery/per_phrase.csv` | `pixi run recovery`, `report-recovery`, then `tables --data results` |
| Fig. 5 LSD | `recovery/per_phrase.csv`, `recovery/random_lsd.csv` | Same recovery/report commands, then `figures --data results` |
| Table II efficiency | `efficiency/*.json` | `pixi run efficiency --output results/efficiency/all.json`, then `tables --data results` |

To rebuild only selected artifacts, the individual generator modules can also
be imported; aggregate figure/table commands expect all their input groups.

The recovery target registry is packaged in `src/data/targets.json`. Its
reference copy here records the original input. The generator uses seed 2029;
main gradient designs use the seed-2028 namespace and State/7-D use the recorded
derived seeds. CSV values are full-precision numerical results; rounding,
ranking and colour choices belong only to the presentation code.
