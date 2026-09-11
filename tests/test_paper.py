from __future__ import annotations

import re
from pathlib import Path


def test_overleaf_tree_is_self_contained() -> None:
    paper = Path(__file__).resolve().parents[1] / "paper"
    source = (paper / "main.tex").read_text(encoding="utf-8")
    assert "../" not in source
    for relative in re.findall(r"\\(?:includegraphics|input)\[[^]]*\]\{([^}]+)\}", source):
        cleaned = relative.replace("\\PaperRoot/", "")
        assert (paper / cleaned).is_file(), relative
    assert (paper / "IEEEtran.cls").is_file()
    assert (paper / "references.bib").is_file()
