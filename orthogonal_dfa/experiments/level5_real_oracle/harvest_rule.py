# pylint: disable=wrong-import-position,wrong-import-order,protected-access,invalid-name,import-error,consider-using-with
"""Replays the round-0 searches of a length_oracles.py run from its family checkpoint, on
the run's own probe streams and selection samples, with the rate-based harvest tests on or
off.  Writes OUT/search_<i>.pkl (as length_oracles.py does) and OUT/restarts.jsonl.

  python -u orthogonal_dfa/experiments/level5_real_oracle/harvest_rule.py RUN_DIR \
      --oracle saw|flat [--no-harvest-tests] --out OUT
"""

import argparse
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import family_ckpt
import numpy as np
from fullloop_mem import VOCAB
from length_oracles import LoggedSearch, oracle

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star import counterexample_synthesis as cs
from orthogonal_dfa.l_star.dfa_utils import held_out_phi, labelled_sample
from orthogonal_dfa.spliceai.load_model import load_spliceai

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--oracle", required=True)
ap.add_argument("--no-harvest-tests", action="store_true")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

base = oracle(a.oracle, default_exon, load_spliceai(400, 0))
ck = family_ckpt.load(f"{a.run}/family_r0.pkl", base, VOCAB)
pst, acc, vs = ck["pst"], ck["acc_threshold"], ck["vs"]
pst.config.counterexample_member_fallback = True
pst.config.counterexample_harvest_tests = not a.no_harvest_tests
cs.CounterexampleSearch = LoggedSearch

# The checkpoint holds restart 0's fresh generator; the selection sample's was spawned
# just before the restarts'.
ss = pst.rng.bit_generator.seed_seq


def sibling(offset):
    key = ss.spawn_key[:-1] + (ss.spawn_key[-1] + offset,)
    return np.random.default_rng(
        np.random.SeedSequence(ss.entropy, spawn_key=key, pool_size=ss.pool_size)
    )


picking = sibling(-1)
halves = [
    labelled_sample(
        pst.oracle, pst.sampler, pst.alphabet_size, cs.SELECTION_HALF, picking
    )
    for _ in range(4)
]
selection = cs.Selection(search=tuple(halves[:2]), pick=tuple(halves[2:]))
streams = [sibling(i) for i in range(pst.config.counterexample_restarts)]
for i, rng in enumerate(streams):
    pst.rng = rng
    t0 = time.time()
    search, ending = cs._counterexample_search(
        pst, vs, acc_threshold=acc, selection=selection.search
    )
    dfa = cs._hypothesis_dfa(pst, search.hypothesis)
    with open(f"{a.out}/search_{i}.pkl", "wb") as f:
        pickle.dump(
            dict(
                ending=ending.kind,
                probes=search.probes,
                best_score=search.best_score,
                harvest=ending.harvest,
                dfa=dfa,
                trajectory=search.trajectory,
            ),
            f,
        )
    row = dict(
        restart=i,
        ending=ending.kind,
        probes=search.probes,
        harvest=len(set(ending.harvest)),
        states=len(dfa.states),
        search_score=search.best_score,
        pick=held_out_phi(dfa, *selection.pick),
        minutes=(time.time() - t0) / 60,
    )
    print(row, flush=True)
    with open(f"{a.out}/restarts.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
