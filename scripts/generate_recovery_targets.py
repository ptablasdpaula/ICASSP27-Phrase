#!/usr/bin/env python3
"""Generate the frozen held-out LHS target registry for phrase recovery."""

from __future__ import annotations

import argparse
import json
from importlib.resources import files

import numpy as np
from icassp27_phrase.paths import OUTPUT as OUTPUT_ROOT
from icassp27_phrase.paths import resolve_path
from scipy.stats import qmc

CARDINALITIES = (1, 2, 4, 6, 8)
COUNT = 150
SEED = 2029

OUTPUT = OUTPUT_ROOT / "targets.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=resolve_path, default=OUTPUT)
    args = parser.parse_args()
    records = []
    for cardinality in CARDINALITIES:
        unit = (
            qmc.LatinHypercube(
                d=2 * cardinality,
                scramble=True,
                optimization=None,
                seed=SEED + cardinality,
            )
            .random(COUNT)
            .reshape(COUNT, cardinality, 2)
        )
        pitch = 80.0 * 4.0 ** unit[..., 0]
        latent_onset = np.sort(unit[..., 1], axis=1)
        available = 1.6 - 0.05 * (cardinality - 1)
        onset = 0.2 + available * latent_onset + 0.05 * np.arange(cardinality)[None]
        for index in range(COUNT):
            records.append(
                {
                    "target_id": f"C{cardinality:02d}-T{index:04d}",
                    "target_index": index,
                    "cardinality": cardinality,
                    "events": [
                        {"f0_hz": float(f0), "onset_seconds": float(time)}
                        for f0, time in zip(pitch[index], onset[index], strict=True)
                    ],
                }
            )
    payload = {
        "schema": "icassp27-phrase-targets-v2",
        "seed": SEED,
        "design": "independent scrambled 2N-dimensional LHS per cardinality",
        "minimum_onset_separation_seconds": 0.05,
        "cardinalities": list(CARDINALITIES),
        "targets_per_cardinality": COUNT,
        "targets": records,
    }
    content = json.dumps(payload, indent=2) + "\n"
    reference = files("icassp27_phrase").joinpath("data/targets.json").read_text()
    if content != reference:
        raise ValueError("Generated target registry differs from the frozen paper input")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)
    print(f"Verified frozen targets: {args.output}")


if __name__ == "__main__":
    main()
