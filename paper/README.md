# Overleaf source

Upload or link this directory as the Overleaf project and select `main.tex` as
the Main document. Alternatively, link the full repository and select
`paper/main.tex`; the source detects both layouts so BibTeX and figure paths
resolve correctly. `IEEEtran.cls`, the bibliography, and every referenced
figure/table are self-contained below `paper/`; no generated build files are
required.

For a local build from the repository root, run `pixi run paper`.
