"""Compare octave fading with the frozen no-fade and linear-Hz controls."""

import csv
import gzip
import json
import shutil
from pathlib import Path

from test_fading_diagonal import signature as linear_signature
from test_octave_fading import DOCS, ROOT, signature

rows = []
for mode, offset in (("plateau", 0), ("fresh", 5)):
    for label, index in (
        ("No fading", offset),
        ("1 s / 1000 Hz", offset + 2),
        ("1 s / 1 octave", None),
    ):
        path = (
            ROOT / f"{mode}.json.gz"
            if index is None
            else Path("results/fading-diagonal") / f"{index:02d}.json.gz"
        )
        with gzip.open(path, "rt") as handle:
            d = json.load(handle)
        assert d["signature"] == (signature() if index is None else linear_signature())
        fit = d["fit"]
        assert abs(fit["best_loss"] - min(s["raw_loss"] for s in fit["trajectory"])) < 1e-12
        rows.append(
            dict(
                mode=mode,
                method=label,
                **{
                    k: d["metrics"][k]
                    for k in (
                        "pitch_mae_cents",
                        "onset_mae_ms",
                        "joint_event_error",
                        "log_spectral_distance_db",
                    )
                },
                canonical_loss=d.get("canonical_loss", d.get("final_canonical_loss")),
                updates=fit["updates"],
                stopped_by=fit["stopped_by"],
            )
        )
        if index is None:
            shutil.copy2(path, DOCS / path.name)
with (DOCS / "summary.csv").open("w") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
lines = [
    "# One-second / one-octave fading",
    "",
    "Complete: C08-T0000, two starting conditions. Earlier matched controls included below.",
    "",
    "[Protocol](protocol.md) · [Qualification](qualification.json) · [Full metrics](summary.csv)",
    "",
    "| Start | Method | Pitch MAE (cents) | Onset MAE (ms) | Updates |",
    "|---|---|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| {r['mode']} | {r['method']} | {r['pitch_mae_cents']:.3f} | "
        f"{r['onset_mae_ms']:.3f} | {r['updates']} |"
    )
lines += [
    "",
    "All four diagonal directions; logarithmic fading, fixed amplitudes, "
    "no Log-Weighing, same registered optimiser/patience and 3000-update cap. "
    "Each fit selects the best own-objective checkpoint. One second / one octave "
    "denotes zero-weight distance, not half-weight distance.",
    "",
    "One phrase only; compare each starting condition separately. "
    "See the CSV for common unfaded loss and LSD. "
    "[Plateau trajectory](plateau.json.gz) · [Fresh trajectory](fresh.json.gz)",
    "",
]
(DOCS / "README.md").write_text("\n".join(lines))
print(json.dumps(rows, indent=2))
