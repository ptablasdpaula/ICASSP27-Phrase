#!/usr/bin/env python3
"""Generate the frozen held-out LHS target registry for phrase recovery."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc

CARDINALITIES = (1, 2, 4, 6, 8)
COUNT = 150
SEED = 2029
OUTPUT = Path("src/data/targets.json")


def main() -> None:
    records = []
    for cardinality in CARDINALITIES:
        unit = qmc.LatinHypercube(
            d=2 * cardinality,
            scramble=True,
            optimization=None,
            seed=SEED + cardinality,
        ).random(COUNT).reshape(COUNT, cardinality, 2)
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
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
