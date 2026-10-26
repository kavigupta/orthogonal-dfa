# pylint: disable=wrong-import-position,wrong-import-order,ungrouped-imports,protected-access,consider-using-with,redefined-outer-name,cell-var-from-loop,global-statement,too-many-locals,invalid-name,missing-function-docstring,import-error,broad-exception-caught,unused-import,reimported,unused-variable,line-too-long
"""Replay a round's counterexample search from a family checkpoint, logging every probe.

  replay.py CKPT [--set field=value ...] [--set k=N] [--probe-seed N] [--tag T]

--set overrides CounterexampleSearchConfig fields (k is patched onto the class);
--probe-seed replaces the pickled rng (omit it to replay the original run exactly).

Writes, next to CKPT, under replay_<tag>/:
  probes.jsonl     one record per probe (outcome, hypothesis size, site counts, oracle use, timing)
  snapshots.jsonl  the hypothesis after every change and every SNAP_EVERY probes: phi on the
                   held-out halves, the tree (midfixes), the learned edges, the DFA's transitions
  snap_<n>.pkl     the same snapshot's DFA and tree, pickled
  summary.json     the ending, and the returned (best-selection-score, closed) hypothesis's numbers
and appends summary.json's record to replays.jsonl.
"""

import argparse
import dataclasses
import faulthandler
import json
import os
import pickle
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

faulthandler.register(signal.SIGUSR1, all_threads=True)

import family_ckpt
from fullloop_mem import VOCAB, held_out, score

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star import counterexample_config as cxcfg
from orthogonal_dfa.l_star import counterexample_search as cxs
from orthogonal_dfa.l_star import counterexample_synthesis as cs
from orthogonal_dfa.l_star.counterexample_probe import (
    EdgeFound,
    End,
    Member,
    Middle,
    Start,
)
from orthogonal_dfa.l_star.examples.gate_composition_residual_len40 import (
    gate_residual_oracle,
)
from orthogonal_dfa.l_star.suffix_family import SuffixFamily
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.oracle import LiftedOracle

SNAP_EVERY = 100

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--probe-seed", type=int)
ap.add_argument("--tag")
ap.add_argument(
    "--restarts",
    type=int,
    help="counterexample_restarts, through _best_search",
)
a = ap.parse_args()

overrides, k = {}, 0
default_cfg = cs.counterexample_search_config(0.98, 0.01)
for kv in a.set:
    key, v = kv.split("=")
    if key == "k":
        k = int(v)
        cxcfg.CounterexampleSearchConfig.k = property(lambda self: k)
    else:
        cur = getattr(default_cfg, key)
        overrides[key] = float(v) if "." in v else type(cur)(v)

tag = a.tag or ("_".join(s.replace("=", "") for s in a.set) or "exact") + (
    f"_p{a.probe_seed}" if a.probe_seed is not None else ""
)
ckdir = os.path.dirname(os.path.abspath(a.ckpt))
outdir = os.path.join(ckdir, f"replay_{tag}")
os.makedirs(outdir, exist_ok=True)
for name in ("probes.jsonl", "snapshots.jsonl", "events.jsonl"):
    open(os.path.join(outdir, name), "w").close()


def write(name, rec):
    with open(os.path.join(outdir, name), "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


base = gate_residual_oracle(
    default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85
)
d = family_ckpt.load(a.ckpt, base, VOCAB)
pst, vs, acc = d["pst"], d["vs"], d["acc_threshold"]
if a.probe_seed is not None:
    pst.rng = np.random.default_rng(a.probe_seed)
# What fullloop_mem.py --plateau sets; checkpoints may predate these fields.
pst.config.counterexample_score_patience = 500
pst.config.counterexample_fnr_harvest = True
if not hasattr(pst.config, "counterexample_restarts"):
    pst.config.counterexample_restarts = 1

halves_path = os.path.join(ckdir, "halves.pkl")
if os.path.exists(halves_path):
    halves = pickle.load(open(halves_path, "rb"))
else:
    halves = held_out(LiftedOracle(base, VOCAB, seed=pst.oracle._seed))
    pickle.dump(halves, open(halves_path, "wb"))

orig_cfg = cs.counterexample_search_config
cs.counterexample_search_config = lambda *x: dataclasses.replace(
    orig_cfg(*x), **overrides
)

# Oracle use: strings sent to the base model (cache misses) and the time spent there.
oracle_use = {"strings": 0, "seconds": 0.0}
orig_mq = LiftedOracle.membership_queries


def counted_mq(self, strings):
    t = time.time()
    out = orig_mq(self, strings)
    oracle_use["strings"] += len(strings)
    oracle_use["seconds"] += time.time() - t
    return out


LiftedOracle.membership_queries = counted_mq

family_for_reads = SuffixFamily(pst, vs)
thresholds = (family_for_reads.reject_thresh, family_for_reads.accept_thresh)


def tree_json(node):
    if isinstance(node, int):
        return node
    midfix, lookup = node
    return {
        "midfix": list(midfix),
        "accept": tree_json(lookup[True]),
        "reject": tree_json(lookup[False]),
    }


def outcome_json(o):
    rec = {"type": type(o).__name__}
    if isinstance(o, (Start, End, Middle)):
        rec["undecided"] = list(o.undecided)
    if isinstance(o, Middle):
        rec["position"] = o.position
    if isinstance(o, Member):
        rec.update(edge=str(o.edge), target=str(o.target), witness=list(o.witness))
    if isinstance(o, EdgeFound):
        rec.update(
            source=str(o.source),
            letter=o.letter,
            target=str(o.target),
            prefix=list(o.prefix),
        )
    return rec


def snapshot(search, n, why):
    hyp = search.hypothesis
    dfa = cs._hypothesis_dfa(pst, hyp)
    phi, top, used = score(dfa, halves)
    rec = dict(
        probe=n,
        why=why,
        version=hyp.version,
        states=hyp.tree.num_states,
        leaves=len(hyp.leaves),
        depth=hyp.tree.depth,
        edges_learned=len(hyp.edges),
        phi=phi,
        top_state_share=top,
        states_used=used,
        tree=tree_json(hyp.tree.root),
        leaf_ids={str(p): hyp.tree.leaf_at(p) for p in hyp.leaves},
        accepting=sorted(hyp.tree.accepting_leaves()),
        edges={str(e): (str(t), list(w)) for e, (t, w) in hyp.edges.items()},
        dfa_initial=dfa.initial_state,
        dfa_transitions={s: dict(t) for s, t in dfa.transitions.items()},
        t=time.time() - t0,
    )
    write("snapshots.jsonl", rec)
    with open(os.path.join(outdir, f"snap_{n:06d}.pkl"), "wb") as f:
        pickle.dump(dict(dfa=dfa, tree=hyp.tree, edges=dict(hyp.edges)), f)
    return rec


def counts_json(c):
    """Every count the round's tests read: a list as (total, distinct), a dict of
    lists as (total, distinct, its worst key and that key's total), else as is."""
    out = {}
    for f in dataclasses.fields(c):
        v = getattr(c, f.name)
        if isinstance(v, list):
            out[f.name] = (len(v), len(set(map(bytes, v))))
        elif isinstance(v, dict):
            flat = (
                [s for xs in v.values() for s in xs]
                if all(isinstance(xs, list) for xs in v.values())
                else []
            )
            worst = max(
                v,
                key=lambda e: len(v[e]) if isinstance(v[e], list) else v[e],
                default=None,
            )
            out[f.name] = (
                len(flat),
                len(set(map(bytes, flat))),
                str(worst),
                (
                    None
                    if worst is None
                    else (len(v[worst]) if isinstance(v[worst], list) else v[worst])
                ),
            )
        else:
            out[f.name] = v
    return out


state = {"n": 0, "last_version": None, "last": None}
orig_step = cxs.CounterexampleSearch.step


def step(self, x):
    t, before = time.time(), dict(oracle_use)
    outcome, charged = self.probe(x)
    probe_s = time.time() - t
    # Re-run the rest of step exactly as the original does, on this probe's results.
    self.probe = lambda _x, _r=(outcome, charged): _r
    try:
        ending = orig_step(self, x)
    finally:
        del self.probe
    state["n"] += 1
    n, hyp, c, cfg = state["n"], self.hypothesis, self.counts, self.config
    rec = dict(
        n=n,
        t=time.time() - t0,
        probe_s=probe_s,
        step_s=time.time() - t,
        x_len=len(x),
        outcome=outcome_json(outcome),
        version=hyp.version,
        leaves=len(hyp.leaves),
        depth=hyp.tree.depth,
        edges_learned=len(hyp.edges),
        since_change=c.probes,
        k=cfg.k,
        counts=counts_json(c),
        oracle_strings=oracle_use["strings"] - before["strings"],
        oracle_s=oracle_use["seconds"] - before["seconds"],
        ending=None if ending is None else ending.kind,
        best_score=getattr(self, "best_score", None),
    )
    write("probes.jsonl", rec)
    if (
        hyp.version != state["last_version"]
        or n % SNAP_EVERY == 0
        or ending is not None
    ):
        why = (
            "end"
            if ending is not None
            else ("change" if hyp.version != state["last_version"] else "periodic")
        )
        state["last"] = snapshot(self, n, why)
        state["last_version"] = hyp.version
    if ending is not None:
        state["ending"] = ending
    return ending


cxs.CounterexampleSearch.step = step

t0 = time.time()
if a.restarts:
    pst.config.counterexample_restarts = a.restarts
selection = cs._selection_sample(pst)
search, ending, _ = cs._best_search(
    pst, vs, acc_threshold=acc, index=0, selection=selection
)
last = snapshot(search, state["n"], "returned")
harvest = sorted(set(map(bytes, ending.harvest)))
family_for_reads.read_many(harvest)
summary = dict(
    ckpt=a.ckpt,
    tag=tag,
    overrides=overrides,
    k=k,
    probe_seed=a.probe_seed,
    ending=ending.kind,
    harvested_edge=str(getattr(ending, "edge", None)),
    probes=state["n"],
    harvest_len=len(ending.harvest),
    harvest_distinct=len(harvest),
    harvest=[
        dict(
            string=list(h),
            length=len(h),
            read=getattr(
                family_for_reads, "_sides", getattr(family_for_reads, "_means", {})
            ).get(h),
        )
        for h in harvest[:20]
    ],
    thresholds=thresholds,
    minutes=(time.time() - t0) / 60,
    selection_score=getattr(search, "best_score", None),
    final_k=search.k,
    best_phi_seen=max(
        r["phi"] for r in map(json.loads, open(os.path.join(outdir, "snapshots.jsonl")))
    ),
    oracle_strings=oracle_use["strings"],
    oracle_minutes=oracle_use["seconds"] / 60,
    **{
        key: last[key]
        for key in (
            "states",
            "leaves",
            "depth",
            "edges_learned",
            "phi",
            "top_state_share",
            "states_used",
        )
    },
)
json.dump(
    summary, open(os.path.join(outdir, "summary.json"), "w"), indent=1, default=str
)
with open(os.path.join(ckdir, "replays.jsonl"), "a") as f:
    f.write(json.dumps(summary, default=str) + "\n")
print("\nREPLAY", json.dumps(summary, default=str), flush=True)
