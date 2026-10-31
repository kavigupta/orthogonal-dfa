"""Adds both oracles' accepts on fresh uniform random strings of each length the
per-length oracle thresholds to extract's JSON, as "random_strings".

    python -m orthogonal_dfa.experiments.length_oracles.random_strings DATA.json

Each length's strings are drawn as median_threshold draws its own, from SEED rather than
the calibration seed; with the calibration seed flat_accepts would be exactly n / 2."""

import json
import sys

import numpy as np

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.spliceai.load_model import load_spliceai

from .oracles import oracle

COUNT = 20000
SEED = 1000


def main():
    path = sys.argv[1]
    model = load_spliceai(400, 0)
    saw, flat = (oracle(kind, default_exon, model) for kind in ("saw", "flat"))
    rows = []
    for L, t in flat.thresholds.items():
        raw = (
            np.random.default_rng(SEED).integers(0, 4, size=(COUNT, L)).astype(np.uint8)
        )
        strings = [row.tobytes() for row in raw]
        rows.append(
            dict(
                length=L,
                n=COUNT,
                threshold=t,
                saw_accepts=int(saw.membership_queries(strings).sum()),
                flat_accepts=int(flat.membership_queries(strings).sum()),
            )
        )
        print(L, rows[-1]["saw_accepts"], rows[-1]["flat_accepts"], flush=True)

    with open(path) as f:
        data = json.load(f)
    data["random_strings"] = rows
    data["random_strings_seed"] = SEED
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
