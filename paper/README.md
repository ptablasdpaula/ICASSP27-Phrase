# Overleaf source

Upload or link this directory as the Overleaf project and select `main.tex` as
the Main document. Alternatively, link the full repository and select
`paper/main.tex`; the source detects both layouts so BibTeX and figure paths
resolve correctly. The supplied ICASSP `spconf.sty` and `IEEEbib.bst`, the
bibliography, and every referenced figure/table are self-contained below
`paper/`; no generated build files are required.

For a local build from the repository root, run `pixi run paper`.

The active manuscript uses the supplied ICASSP article template with default
spacing and 10-point body text. `real_template/` remains the original template
archive; `IEEEtran.cls` is retained only for the historical reviewed manuscript.
