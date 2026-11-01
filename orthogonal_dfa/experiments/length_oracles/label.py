"""Both oracles' labels on the same fresh strings, lifted with seed 0.

    python -u -m orthogonal_dfa.experiments.length_oracles.label

fills the cache with both oracles' fits and labels."""

import numpy as np
from permacache import permacache

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.oracle import LiftedOracle
from orthogonal_dfa.superlanguage.sampler import SuperSampler

from .oracles import SYMBOLS, VOCAB, oracle

SEED = 78
COUNT = 200_000
CHUNK = 20_000


def strings(n):
    rng = np.random.default_rng(SEED)
    sampler = SuperSampler(VOCAB, SYMBOLS)
    return [tuple(sampler.sample(rng, VOCAB.alphabet_size)) for _ in range(n)]


@permacache("orthogonal_dfa/experiments/length_oracles/label/labels")
def labels(kind, n):
    """0/1 labels of ``strings(n)``."""
    s = strings(n)
    lifted = LiftedOracle(
        oracle(kind, default_exon, load_spliceai(400, 0)), VOCAB, seed=0
    )
    return np.concatenate(
        [lifted.membership_queries(s[i : i + CHUNK]) for i in range(0, n, CHUNK)]
    ).astype(np.int8)


def main():
    for kind in ("saw", "flat"):
        print(kind, "accept rate", labels(kind, COUNT).mean(), flush=True)


if __name__ == "__main__":
    main()
