# pylint: disable=protected-access
"""Adds both oracles' accepts on fresh uniform random strings of each length 35-84 to
extract's JSON, as "random_strings".

    python -m orthogonal_dfa.experiments.length_oracles.random_strings DATA.json

Each length's strings are drawn as median_threshold draws its own, from SEED rather than
the calibration seed; with the calibration seed flat_accepts would be exactly n / 2."""

import json
import sys

import numpy as np

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import (
    fit_gate_composition_residual,
    gate_residual_oracle,
)
from orthogonal_dfa.l_star.examples.spliceai_oracle import (
    flanks,
    median_threshold,
    run_over_middles,
)
from orthogonal_dfa.spliceai.exon_score import SpliceAIExonScore, device_of
from orthogonal_dfa.spliceai.load_model import load_spliceai

from .oracles import LEN_HI, LEN_LO, SAW_LENGTH

COUNT = 20000
SEED = 1000
CHUNK = 1024


def main():
    path = sys.argv[1]
    model = load_spliceai(400, 0)
    gscore = fit_gate_composition_residual(
        SpliceAIExonScore(model).eval(), default_exon, len_lo=LEN_LO, len_hi=LEN_HI
    )
    # The fit's band ends before LEN_HI.
    thresholds = {
        L: median_threshold(gscore, default_exon, L) for L in range(LEN_LO, LEN_HI)
    }
    saw = gate_residual_oracle(
        default_exon, model, length=SAW_LENGTH, len_lo=LEN_LO, len_hi=LEN_HI
    )
    assert saw._threshold == thresholds[SAW_LENGTH], (
        saw._threshold,
        thresholds[SAW_LENGTH],
    )
    flank_l, flank_r = flanks(default_exon)
    rows = []
    for L, t in thresholds.items():
        raw = (
            np.random.default_rng(SEED).integers(0, 4, size=(COUNT, L)).astype(np.uint8)
        )
        scores = run_over_middles(
            gscore,
            flank_l,
            flank_r,
            [row.tobytes() for row in raw],
            device=device_of(gscore),
            chunk=CHUNK,
        )
        rows.append(
            dict(
                length=L,
                n=len(scores),
                threshold=t,
                saw_accepts=int((scores > thresholds[SAW_LENGTH]).sum()),
                flat_accepts=int((scores > t).sum()),
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
