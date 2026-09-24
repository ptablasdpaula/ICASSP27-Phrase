"""Portable paths and source provenance for experiment commands."""

from __future__ import annotations

import hashlib
import json
import os
from importlib.metadata import distribution
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env", override=False)


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


OUTPUT = resolve_path(os.getenv("PHRASE_OUTPUT_DIR", "results"))
CACHE = resolve_path(os.getenv("PHRASE_CACHE_DIR", "results/cache"))
LOGS = resolve_path(os.getenv("PHRASE_LOG_DIR", "results/logs"))
REFERENCE = REPO / "paper" / "results"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(CACHE / "numba"))


def scientific_signature() -> tuple[str, dict[str, str]]:
    paths = [
        *sorted((REPO / "src").rglob("*.py")),
        *sorted((REPO / "scripts").glob("*.py")),
        REPO / "src/data/targets.json",
        REPO / "pyproject.toml",
        REPO / "pixi.lock",
    ]
    hashes = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    # Keep resume signatures sensitive to the installed loss implementation,
    # independently of its site-packages location or installation method.
    cels = distribution("cels-audio")
    hashes["dependencies/cels-audio/version"] = hashlib.sha256(cels.version.encode()).hexdigest()
    sources = [f for f in cels.files or () if str(f).startswith("cels/") and str(f).endswith(".py")]
    if not sources:
        raise RuntimeError(
            "Install the published cels-audio package to record its source signature"
        )
    for source in sorted(sources):
        hashes[f"dependencies/cels-audio/{source}"] = hashlib.sha256(
            Path(cels.locate_file(source)).read_bytes()
        ).hexdigest()
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), hashes


def validate_reference(root: Path = REFERENCE) -> None:
    """Validate archived data against its original manifest, not today's source."""
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Reference result missing or modified: {path}")
