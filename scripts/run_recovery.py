"""Run the nine reported objectives with the shared independent-phrase optimiser."""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import time
from dataclasses import asdict

import torch
from icassp27_phrase.data import load_target
from icassp27_phrase.metrics import recovery_metrics
from icassp27_phrase.optimization import SCHEDULE
from icassp27_phrase.optimization import fit_batch as fit
from icassp27_phrase.paths import OUTPUT, REPO, resolve_path
from icassp27_phrase.paths import scientific_signature as signature
from icassp27_phrase.runtime import configure_reproducibility, require_df2_backend
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.synth.config import CARDINALITIES

ROOT = OUTPUT / "recovery"
LOSSES = (
    "single_stft",
    "smooth_mss",
    "sot_published_composite",
    "linear_jtfot",
    "log_jtfot",
    "cel",
    "log_cel",
    "dec_cel",
    "tlog_cel",
)
LABELS = ("SS", "SmoMSS", "SOT", "TFW2", "logTFW2", "CeL", "logCeL", "decCeL", "tlogCeL")
TARGETS_PER_CELL = 150
SHARD_SPECS = tuple(
    (loss, n, start, size)
    for loss in LOSSES
    for n in CARDINALITIES
    for size in [50 if loss == "sot_published_composite" else 75]
    for start in range(0, TARGETS_PER_CELL, size)
)


def shard_coordinates(shard):
    return SHARD_SPECS[shard]


def run(shard, device="cuda"):
    name, cardinality, begin, target_count = shard_coordinates(shard)
    sig, hashes = signature()
    outpath = (
        ROOT
        / "raw"
        / name
        / f"C{cardinality:02d}-T{begin:04d}-{begin + target_count - 1:04d}.json.gz"
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    if outpath.exists():
        with gzip.open(outpath, "rt") as f:
            old = json.load(f)
        if (
            old["signature"] != sig
            or old["loss"] != name
            or old["cardinality"] != cardinality
            or old["target_range"] != [begin, begin + target_count]
            or len(old["rows"]) != target_count
            or old["schedule"] != json.loads(json.dumps(asdict(SCHEDULE)))
        ):
            raise ValueError(f"stale shard {outpath}")
        print("EXISTS", outpath, flush=True)
        return
    synth = PhraseSynth().to(device)
    metadata, phrases = zip(
        *(
            load_target(cardinality, i + 1, device=device)
            for i in range(begin, begin + target_count)
        ),
        strict=True,
    )
    # Each batch groups independent fits with their own Adam and scheduler state.
    with torch.no_grad():
        target_audio = synth(
            torch.stack([p.f0_hz for p in phrases]),
            torch.stack([p.onset_seconds for p in phrases]),
        )
    started = time.perf_counter()
    f0, onset, final, strict_best, initial, updates, reductions, lr_events = fit(
        target_audio, cardinality, name, synth
    )
    rows = []
    for i, meta in enumerate(metadata):
        metrics = recovery_metrics(
            f0[i].cpu().numpy(), onset[i].cpu().numpy(), meta.f0_hz, meta.onset_seconds
        )
        rows.append(
            {
                "target_id": meta.target_id,
                "target": asdict(meta),
                "reported_f0_hz": f0[i].cpu().tolist(),
                "reported_onset_seconds": onset[i].cpu().tolist(),
                "initial_loss": float(initial[i]),
                "reported_loss": float(strict_best[i]),
                "terminal_loss": float(final[i]),
                "updates": int(updates[i]),
                "stopped_by": (
                    "maximum_updates" if updates[i] >= SCHEDULE.maximum_updates else "patience"
                ),
                "lr_reductions": int(reductions[i]),
                "lr_events": lr_events[i],
                "metrics": metrics,
            }
        )
    payload = {
        "schema": "phrase-recovery-16k-v1",
        "signature": sig,
        "source_hashes": hashes,
        "source_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "shard": shard,
        "loss": name,
        "label": LABELS[LOSSES.index(name)],
        "cardinality": cardinality,
        "target_range": [begin, begin + target_count],
        "schedule": asdict(SCHEDULE),
        "reported_iterate": "strict lowest-loss iterate",
        "matching": "squared octaves plus squared seconds; one octave equals one second",
        "target_minimum_separation_seconds": 0.05,
        "renderer": synth.provenance(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "wall_seconds": time.perf_counter() - started,
        "rows": rows,
    }
    temporary = outpath.with_suffix(".tmp.gz")
    with gzip.open(temporary, "wt") as f:
        json.dump(payload, f, separators=(",", ":"), allow_nan=False)
    temporary.replace(outpath)
    print(
        "COMPLETE", shard, name, cardinality, begin, round(payload["wall_seconds"], 2), flush=True
    )


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", type=int, help=f"One array task, 0..{len(SHARD_SPECS) - 1}; omit for all"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=resolve_path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.output
    if args.task is not None and not 0 <= args.task < len(SHARD_SPECS):
        parser.error("invalid task index")
    configure_reproducibility()
    torch.set_num_threads(1)
    require_df2_backend(args.device)
    for task in range(len(SHARD_SPECS)) if args.task is None else (args.task,):
        run(task, args.device)


if __name__ == "__main__":
    main()
