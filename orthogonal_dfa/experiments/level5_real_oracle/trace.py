# pylint: disable=wrong-import-position,wrong-import-order,protected-access,consider-using-with,invalid-name,missing-function-docstring,import-error,line-too-long,unsubscriptable-object,duplicate-code,redefined-outer-name,attribute-defined-outside-init
"""Replay one restart of a round's counterexample search from a family checkpoint,
logging every change to the hypothesis.

  trace.py CKPT --restart I --out DIR

Restart I runs on the same probe stream as restart I of the run that wrote CKPT
(for I below its restart count), and every restart is scored on the same
selection sample.  Writes, under DIR:

  events.jsonl   one line per probe that changed the hypothesis or its walk start,
                 and per score: what changed, the hypothesis's size, its
                 selection score and its phi on the 1111/2222 halves
  snapshots.pkl  (probe, (tree, edges, leaves)) after each change
  final.json     the ending, the returned hypothesis's scores, and timings
  final.pkl      the returned hypothesis's snapshot and DFA, before and after closing
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
from orthogonal_dfa.l_star.counterexample_probe import EdgeFound, Member
from orthogonal_dfa.l_star.counterexample_search import CounterexampleSearch
from orthogonal_dfa.l_star.dfa_utils import held_out_phi, labelled_sample
from orthogonal_dfa.l_star.examples.gate_composition_residual import (
    gate_residual_oracle,
)
from orthogonal_dfa.l_star.fallbacks.score_patience import (
    ScorePatience,
    restore,
    snapshot,
)
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.oracle import LiftedOracle

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--restart", type=int, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

base = gate_residual_oracle(
    default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85
)
ck = family_ckpt.load(a.ckpt, base, VOCAB)
pst, acc, vs = ck["pst"], ck["acc_threshold"], ck["vs"]

halves_path = os.path.join(os.path.dirname(os.path.abspath(a.ckpt)), "halves.pkl")
if os.path.exists(halves_path):
    halves = pickle.load(open(halves_path, "rb"))
else:
    halves = held_out(LiftedOracle(base, VOCAB, seed=pst.oracle._seed))

# The checkpoint holds restart 0's generator; see cross.py.
ss = pst.rng.bit_generator.seed_seq


def sibling(offset):
    key = ss.spawn_key[:-1] + (ss.spawn_key[-1] + offset,)
    return np.random.default_rng(
        np.random.SeedSequence(ss.entropy, spawn_key=key, pool_size=ss.pool_size)
    )


t_sel = time.time()
# The selection sample of the code fl9 ran, which drew it from a fixed generator.
strings, labels = labelled_sample(
    pst.oracle,
    pst.sampler,
    pst.alphabet_size,
    2 * cs.SELECTION_HALF,
    np.random.default_rng(0x5C0E2),
)
half = cs.SELECTION_HALF
selection = ((strings[:half], labels[:half]), (strings[half:], labels[half:]))
sel_minutes = (time.time() - t_sel) / 60
pst.rng = sibling(a.restart)

log = open(os.path.join(a.out, "events.jsonl"), "w")
snapshots = []
t0 = time.time()


def emit(**row):
    row["minutes"] = (time.time() - t0) / 60
    log.write(json.dumps(row, default=str) + "\n")
    log.flush()


def path_str(p):
    return "".join("1" if b else "0" for b in p)


def describe(outcome):
    if isinstance(outcome, Member):
        return dict(
            kind="edge",
            source=path_str(outcome.edge[0]),
            letter=outcome.edge[1],
            target=path_str(outcome.target),
            witness=outcome.witness.hex(),
        )
    if isinstance(outcome, EdgeFound):
        return dict(
            kind="split",
            source=path_str(outcome.source),
            letter=outcome.letter,
            target=path_str(outcome.target),
            prefix=outcome.prefix.hex(),
        )
    return dict(kind=type(outcome).__name__)


class Traced(CounterexampleSearch):
    def probe(self, x):
        outcome, charged = super().probe(x)
        self._last = (x, outcome)
        return outcome, charged

    def _split(self, found):
        before = self.hypothesis.tree.num_states
        super()._split(found)
        self._split_info = dict(states_before=before)

    def step(self, x):
        version, k = self.hypothesis.version, self.k
        self._split_info = {}
        ending = super().step(x)
        if self.hypothesis.version != version or self.k != k:
            h = self.hypothesis
            dfa = cs._hypothesis_dfa(pst, h)
            phi, top, used = score(dfa, halves)
            emit(
                event="change",
                probe=self.probes,
                x=x.hex(),
                k=self.k,
                deepened=self.k != k,
                **describe(self._last[1]),
                **self._split_info,
                states=h.tree.num_states,
                edges=len(h.edges),
                phi=phi,
                top_state_share=top,
                states_used=used,
            )
            snapshots.append((self.probes, snapshot(h)))
        if ending is not None:
            emit(
                event="ending",
                probe=self.probes,
                ending=ending.kind,
                harvest=[s.hex() for s in ending.harvest],
            )
        return ending


orig_observe = ScorePatience.observe


def observe(self, search):
    if search.hypothesis.version == self._scored_version:
        return orig_observe(self, search)
    best, values, score_fn = self.best_score, [], self.score
    self.score = lambda hypothesis: values.append(score_fn(hypothesis)) or values[-1]
    try:
        ending = orig_observe(self, search)
    finally:
        self.score = score_fn
    emit(
        event="score",
        probe=search.probes,
        version=search.hypothesis.version,
        selection_score=values[0],
        new_best=self.best_score != best,
    )
    return ending


ScorePatience.observe = observe

cs.CounterexampleSearch = Traced
pst.config.counterexample_skip_undecided = True
search, ending = cs._counterexample_search(
    pst, vs, acc_threshold=acc, selection=selection
)
search_minutes = (time.time() - t0) / 60
hyp = search.hypothesis
dfa = cs._hypothesis_dfa(pst, hyp)
phi, top, used = score(dfa, halves)
patience = search.score_patience
best_snap = getattr(patience, "_best", None)
unclosed_dfa = None
if best_snap is not None:
    final = snapshot(search.hypothesis)
    restore(search.hypothesis, best_snap)
    unclosed_dfa = cs._hypothesis_dfa(pst, search.hypothesis)
    restore(search.hypothesis, final)
summary = dict(
    ckpt=a.ckpt,
    restart=a.restart,
    ending=ending.kind,
    harvest_size=len(ending.harvest),
    probes=search.probes,
    k=search.k,
    max_k=search.max_k,
    best_at=getattr(patience, "_best_at", 0),
    best_score=getattr(patience, "best_score", None),
    states=hyp.tree.num_states,
    edges=len(hyp.edges),
    selection_score=held_out_phi(dfa, *selection),
    phi=phi,
    top_state_share=top,
    states_used=used,
    unclosed_phi=None if unclosed_dfa is None else score(unclosed_dfa, halves)[0],
    selection_minutes=sel_minutes,
    search_minutes=search_minutes,
)
json.dump(summary, open(os.path.join(a.out, "final.json"), "w"), indent=1, default=str)
with open(os.path.join(a.out, "snapshots.pkl"), "wb") as f:
    pickle.dump(snapshots, f)
with open(os.path.join(a.out, "final.pkl"), "wb") as f:
    pickle.dump(
        dict(
            snapshot=snapshot(search.hypothesis),
            best_snapshot=best_snap,
            dfa=dfa,
            unclosed_dfa=unclosed_dfa,
            harvest=ending.harvest,
            undecided=list(getattr(patience, "undecided", ())),
        ),
        f,
    )
print("SUMMARY", json.dumps(summary, default=str), flush=True)
