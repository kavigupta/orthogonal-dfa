"""Both oracles' labels on the same fresh strings, lifted with seed 0.

    python -u -m orthogonal_dfa.experiments.length_oracles.label OUT.npz [N]

OUT.npz holds s (N x 36 symbols), saw and flat (0/1 labels)."""

import sys

import numpy as np

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.oracle import LiftedOracle
from orthogonal_dfa.superlanguage.sampler import SuperSampler

from .oracles import SYMBOLS, VOCAB, oracle

SEED = 78
CHUNK = 20_000


def main():
    out = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
    rng = np.random.default_rng(SEED)
    sampler = SuperSampler(VOCAB, SYMBOLS)
    strings = [tuple(sampler.sample(rng, VOCAB.alphabet_size)) for _ in range(n)]
    model = load_spliceai(400, 0)
    labels = {}
    for kind in ("saw", "flat"):
        lifted = LiftedOracle(oracle(kind, default_exon, model), VOCAB, seed=0)
        labels[kind] = np.concatenate(
            [
                lifted.membership_queries(strings[i : i + CHUNK])
                for i in range(0, n, CHUNK)
            ]
        ).astype(np.int8)
        print(kind, "accept rate", labels[kind].mean(), flush=True)
    np.savez(out, s=np.array(strings, dtype=np.int8), **labels)


if __name__ == "__main__":
    main()
