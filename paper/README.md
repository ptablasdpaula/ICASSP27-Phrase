# Manuscript

Select `paper/main.tex` as the main document when linking this repository to
Overleaf, or `main.tex` when uploading this directory alone. Build locally with
`pixi run paper`. All referenced assets and template files are retained here.

`figures/` contains the submitted assets. `results/` contains the compact
numerical inputs and original provenance needed to regenerate plots and tables
without rerunning optimisation. Generators live in the repository's `scripts/`;
see the root README for commands.

`IEEEbib-initials.bst` derives from the supplied `IEEEbib.bst`, using author
initials and et al. for seven or more authors. The original style is retained.
The manuscript and IEEE template material are excluded from the code's MIT licence.
