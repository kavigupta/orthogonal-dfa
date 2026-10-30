# pylint: disable=wrong-import-position,import-error,invalid-name
"""Both oracles' labels on the same fresh 36-symbol strings (wildcards lifted with seed 0).

  python -u orthogonal_dfa/experiments/level5_real_oracle/length_oracles_label.py OUT.npz [N]

OUT.npz holds s (N x 36 symbols), saw and flat (0/1 labels)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fullloop_mem import VOCAB
from length_oracles import oracle

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.oracle import LiftedOracle
from orthogonal_dfa.superlanguage.sampler import SuperSampler

out = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
rng = np.random.default_rng(78)
sampler = SuperSampler(VOCAB, 36)
strings = [tuple(sampler.sample(rng, VOCAB.alphabet_size)) for _ in range(n)]
model = load_spliceai(400, 0)
labels = {}
for kind in ("saw", "flat"):
    lifted = LiftedOracle(oracle(kind, default_exon, model), VOCAB, seed=0)
    labels[kind] = np.concatenate(
        [
            lifted.membership_queries(strings[i : i + 20_000])
            for i in range(0, n, 20_000)
        ]
    ).astype(np.int8)
    print(kind, "accept rate", labels[kind].mean(), flush=True)
np.savez(out, s=np.array(strings, dtype=np.int8), **labels)
