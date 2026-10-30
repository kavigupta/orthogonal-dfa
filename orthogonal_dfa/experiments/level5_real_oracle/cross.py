# pylint: disable=wrong-import-position,wrong-import-order,protected-access,consider-using-with,invalid-name,missing-function-docstring,import-error,line-too-long,unsubscriptable-object,duplicate-code
"""Run one round's counterexample search on the pool of one family checkpoint and
the family (suffixes and decision boundary) of another.

  cross.py POOL_CKPT FAMILY_CKPT --out DIR [--restarts K] [--probe-seed N] [--patience P] [--no-members]

--no-members turns off the member fallback, so the search neither reads stuck edges
on members nor closes from them.

Writes DIR/summary.json: the family's FNR on the pool's representative prefixes,
and the returned DFA's held-out phi on the 1111/2222 halves.
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
from fullloop_mem import VOCAB, held_out, score

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star import counterexample_synthesis as cs
from orthogonal_dfa.l_star.dfa_utils import held_out_phi, labelled_sample
from orthogonal_dfa.l_star.examples.gate_composition_residual import (
    gate_residual_oracle,
)
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.oracle import LiftedOracle

ap = argparse.ArgumentParser()
ap.add_argument("pool")
ap.add_argument("family")
ap.add_argument("--out", required=True)
ap.add_argument("--restarts", type=int, default=3)
ap.add_argument("--probe-seed", type=int)
ap.add_argument("--patience", type=int)
ap.add_argument("--no-members", action="store_true")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

base = gate_residual_oracle(
    default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85
)
pool = family_ckpt.load(a.pool, base, VOCAB)
fam = family_ckpt.load(a.family, base, VOCAB)
pst, acc = pool["pst"], pool["acc_threshold"]
suffixes = [fam["pst"].table.suffix(i) for i in fam["vs"]]
vs = [pst.table.intern_suffix(s) for s in suffixes]
pst.decision_boundary = fam["pst"].decision_boundary
if a.probe_seed is not None:
    pst.rng = np.random.default_rng(
        np.random.SeedSequence(a.probe_seed, spawn_key=(0,))
    )
if a.patience is not None:
    pst.config.counterexample_score_patience = a.patience
pst.config.counterexample_restarts = a.restarts
pst.config.counterexample_member_fallback = not a.no_members

t0 = time.time()
fnr, worst = pst.fnr_from_decision(pst.compute_decision(vs, pst.table.representative))
print(f"family FNR on pool's representative prefixes: {fnr:.4f}", flush=True)
fnr_min = (time.time() - t0) / 60

halves_path = os.path.join(os.path.dirname(os.path.abspath(a.pool)), "halves.pkl")
if os.path.exists(halves_path):
    halves = pickle.load(open(halves_path, "rb"))
else:
    halves = held_out(LiftedOracle(base, VOCAB, seed=pst.oracle._seed))
    pickle.dump(halves, open(halves_path, "wb"))

t1 = time.time()
# The checkpoint holds restart 0's generator: harvest_diag dumps inside
# _counterexample_search, after _best_search spawned the restarts from the run's
# stream, which _selection_sample had spawned the selection sample's stream from
# just before.  Rebuild those rather than spawning again.
ss = pst.rng.bit_generator.seed_seq


def sibling(offset):
    key = ss.spawn_key[:-1] + (ss.spawn_key[-1] + offset,)
    return np.random.default_rng(
        np.random.SeedSequence(ss.entropy, spawn_key=key, pool_size=ss.pool_size)
    )


streams = [sibling(i) for i in range(a.restarts)]
picking = sibling(-1)
selection = tuple(
    labelled_sample(
        pst.oracle, pst.sampler, pst.alphabet_size, cs.SELECTION_HALF, picking
    )
    for _ in range(2)
)
restarts, best = [], None
for i, rng in enumerate(streams):
    pst.rng = rng
    search, ending = cs._counterexample_search(
        pst, vs, acc_threshold=acc, selection=selection
    )
    hyp = search.hypothesis
    dfa = cs._hypothesis_dfa(pst, hyp)
    sel = held_out_phi(dfa, *selection)
    restarts.append(
        dict(
            selection_score=sel,
            phi=score(dfa, halves)[0],
            states=hyp.tree.num_states,
            ending=ending.kind,
        )
    )
    print(f"restart {i}: {restarts[-1]}", flush=True)
    if best is None or sel > best[0]:
        best = (sel, search, ending, hyp, dfa)
_, search, ending, hyp, dfa = best
phi, top, used = score(dfa, halves)
summary = dict(
    pool=a.pool,
    family=a.family,
    restarts=a.restarts,
    no_members=a.no_members,
    probe_seed=a.probe_seed,
    family_size=len(vs),
    decision_boundary=pst.decision_boundary,
    family_fnr=fnr,
    worst_population=str(worst),
    representative_prefixes=int(np.sum(pst.table.representative)),
    ending=ending.kind,
    states=hyp.tree.num_states,
    selection_score=held_out_phi(dfa, *selection),
    per_restart=restarts,
    phi=phi,
    top_state_share=top,
    states_used=used,
    fnr_minutes=fnr_min,
    search_minutes=(time.time() - t1) / 60,
)
json.dump(
    summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=str
)
with open(os.path.join(a.out, "dfa.pkl"), "wb") as f:
    pickle.dump(dfa, f)
print("SUMMARY", json.dumps(summary, default=str), flush=True)
