"""Compute recovery metrics and LSD from completed independent-phrase fits."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from _recovery_plot import REPORT_LOSSES, random_derangement
from icassp27_phrase.data import load_target
from icassp27_phrase.metrics import log_spectral_distance, recovery_metrics
from icassp27_phrase.optimization import SCHEDULE
from icassp27_phrase.paths import OUTPUT, resolve_path, scientific_signature
from icassp27_phrase.runtime import configure_reproducibility, require_df2_backend
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.synth.config import CARDINALITIES


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def report(root: Path, device: str):
    records = {}
    signature, hashes = scientific_signature()
    for path in sorted((root / "raw").glob("*/*.json.gz")):
        with gzip.open(path, "rt") as stream:
            data = json.load(stream)
        if data["signature"] != signature or data["schedule"] != json.loads(
            json.dumps(asdict(SCHEDULE))
        ):
            raise ValueError(f"Incompatible recovery shard: {path}")
        loss, n = data["loss"], data["cardinality"]
        for row in data["rows"]:
            index = row["target"]["index"] - 1
            key = loss, n, index
            if key in records:
                raise ValueError(f"Duplicate fit: {key}")
            meta, _ = load_target(n, index + 1)
            if (
                row["target_id"] != meta.target_id
                or row["target"]["f0_hz"] != list(meta.f0_hz)
                or row["target"]["onset_seconds"] != list(meta.onset_seconds)
            ):
                raise ValueError(f"Target mismatch: {key}")
            records[key] = row
    expected = {(loss, n, i) for loss in REPORT_LOSSES for n in CARDINALITIES for i in range(150)}
    if set(records) != expected:
        raise ValueError(
            f"Expected 6750 fits: {len(expected - set(records))} missing, "
            f"{len(set(records) - expected)} unexpected"
        )
    configure_reproducibility()
    torch.set_num_threads(1)
    require_df2_backend(device)
    synth = PhraseSynth().to(device)
    rows = []
    random_rows = []
    with torch.no_grad():
        for n in CARDINALITIES:
            targets = [load_target(n, i + 1, device=device)[1] for i in range(150)]
            f = torch.stack([p.f0_hz for p in targets])
            t = torch.stack([p.onset_seconds for p in targets])
            # Render in small batches, independent of the original optimisation batch size.
            audio = torch.cat([synth(f[i : i + 10], t[i : i + 10]) for i in range(0, 150, 10)])
            permutation = random_derangement(n)
            for begin in range(0, 150, 10):
                end = begin + 10
                random_lsd = log_spectral_distance(audio[permutation[begin:end]], audio[begin:end])
                random_rows.extend(
                    {
                        "cardinality": n,
                        "target_index": i,
                        "log_spectral_distance_db": float(random_lsd[i - begin]),
                    }
                    for i in range(begin, end)
                )
                for loss in REPORT_LOSSES:
                    batch = [records[loss, n, i] for i in range(begin, end)]
                    cf = torch.tensor(
                        [r["reported_f0_hz"] for r in batch], dtype=torch.float64, device=device
                    )
                    ct = torch.tensor(
                        [r["reported_onset_seconds"] for r in batch],
                        dtype=torch.float64,
                        device=device,
                    )
                    distances = log_spectral_distance(synth(cf, ct), audio[begin:end])
                    for j, r in enumerate(batch):
                        metric = recovery_metrics(
                            cf[j].cpu().numpy(),
                            ct[j].cpu().numpy(),
                            f[begin + j].cpu().numpy(),
                            t[begin + j].cpu().numpy(),
                        )
                        rows.append(
                            {
                                "loss": loss,
                                "cardinality": n,
                                "target_index": begin + j,
                                "target_id": r["target_id"],
                                "pitch_mae_cents": metric["pitch_mae_cents"],
                                "onset_mae_ms": metric["onset_mae_ms"],
                                "log_spectral_distance_db": float(distances[j]),
                                **{
                                    k: r[k]
                                    for k in (
                                        "initial_loss",
                                        "reported_loss",
                                        "terminal_loss",
                                        "updates",
                                        "stopped_by",
                                        "lr_reductions",
                                    )
                                },
                                "target_f0_hz": json.dumps(r["target"]["f0_hz"]),
                                "target_onset_seconds": json.dumps(r["target"]["onset_seconds"]),
                                "reported_f0_hz": json.dumps(r["reported_f0_hz"]),
                                "reported_onset_seconds": json.dumps(r["reported_onset_seconds"]),
                            }
                        )
            print(f"Reported {n} events", flush=True)
    write_csv(root / "per_phrase.csv", rows)
    write_csv(root / "random_lsd.csv", random_rows)
    summary = []
    for loss in REPORT_LOSSES:
        for n in CARDINALITIES:
            selected = [r for r in rows if r["loss"] == loss and r["cardinality"] == n]
            summary.append(
                {
                    "loss": loss,
                    "cardinality": n,
                    **{
                        m: float(np.median([r[m] for r in selected]))
                        for m in ("pitch_mae_cents", "onset_mae_ms", "log_spectral_distance_db")
                    },
                }
            )
    write_csv(root / "summary.csv", summary)
    (root / "provenance.json").write_text(
        json.dumps(
            {
                "signature": signature,
                "source_hashes": hashes,
                "schedule": asdict(SCHEDULE),
                "fit_count": len(rows),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=resolve_path, default=OUTPUT / "recovery")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    a = p.parse_args()
    report(a.root, a.device)
