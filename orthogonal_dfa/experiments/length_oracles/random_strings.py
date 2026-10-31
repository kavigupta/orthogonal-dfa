"""Both oracles' accepts on fresh uniform random strings of each length the per-length
oracle thresholds, written to ``data/neural/length_oracles/random_strings.json``.

    python -m orthogonal_dfa.experiments.length_oracles.random_strings

Each length's strings are drawn as median_threshold draws its own, from SEED rather than
the calibration seed; with the calibration seed flat_accepts would be exactly n / 2."""

import numpy as np

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.spliceai.load_model import load_spliceai

from ..capal_comparison.core import REPO_ROOT
from ..random_minimal.run import write_json
from .oracles import oracle

COUNT = 20000
SEED = 1000
OUT = REPO_ROOT / "data" / "neural" / "length_oracles" / "random_strings.json"


def main():
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
    write_json(OUT, dict(seed=SEED, rows=rows))


if __name__ == "__main__":
    main()
