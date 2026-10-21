# pylint: disable=wrong-import-position,wrong-import-order,ungrouped-imports,protected-access,consider-using-with,redefined-outer-name,cell-var-from-loop,global-statement,too-many-locals,invalid-name,missing-function-docstring,import-error,broad-exception-caught,unused-import,reimported,unused-variable,line-too-long
"""Full synthesis loop on the len40 real oracle; held-out phi of every round's DFA.

Run from inside the worktree under test (PYTHONPATH = that worktree)."""

import argparse
import json
import os
import pickle
import subprocess
import threading
import time

import numpy as np

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.counterexample_synthesis import (
    counterexample_driven_synthesis,
)
from orthogonal_dfa.l_star.examples.gate_composition_residual import (
    gate_residual_oracle,
)
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.tracker import SynthesisTracker
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.oracle import LiftedOracle
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

VOCAB = KmerVocabulary(kmers=((3, 0, 2), (3, 0, 0), (3, 2, 0)), base_alphabet_size=4)
N_HALF = 6000


def held_out(oracle):
    halves = []
    for sd in (1111, 2222):
        rng = np.random.default_rng(sd)
        samp = SuperSampler(VOCAB, 36)
        s = [samp.sample(rng, VOCAB.alphabet_size) for _ in range(N_HALF)]
        halves.append((s, np.asarray(oracle.membership_queries(s), dtype=float)))
    return halves


def score(dfa, halves):
    def walk(w):
        st = dfa.initial_state
        for c in w:
            st = dfa.transitions[st][c]
        return st

    (sA, yA), (sB, yB) = halves
    rA = np.array([walk(w) for w in sA])
    rB = np.array([walk(w) for w in sB])
    rate = {s: yA[rA == s].mean() for s in np.unique(rA)}
    pred = np.array([rate.get(s, yA.mean()) for s in rB])
    phi = 0.0 if pred.std() == 0 else abs(float(np.corrcoef(pred, yB)[0, 1]))
    _, counts = np.unique(rB, return_counts=True)
    return phi, float(counts.max() / len(rB)), len(np.unique(rB))


class Tracker(SynthesisTracker):
    def __init__(self, halves, out):
        self.halves, self.out, self.rows, self.t0 = halves, out, {}, time.time()

    def on_search_ended(self, ending, round_index):
        self.rows.setdefault(round_index, {})["ending"] = ending.kind

    def on_initial_dfa_found(self, dfa, tree, round_index):
        phi, top, used = score(dfa, self.halves)
        self.rows.setdefault(round_index, {}).update(
            states=len(dfa.states),
            phi=phi,
            top_state_share=top,
            states_used=used,
            elapsed_h=(time.time() - self.t0) / 3600,
        )
        with open(f"{self.out}/round{round_index}.pkl", "wb") as f:
            pickle.dump(dfa, f)

    def on_consistency_estimated(self, consistency, round_index):
        self.rows[round_index]["consistency"] = consistency
        self.flush()

    def on_certificate_decided(self, certified, round_index):
        self.rows[round_index]["certified"] = certified
        self.flush()

    def flush(self):
        with open(f"{self.out}/rounds.json", "w") as f:
            json.dump(self.rows, f, indent=1)


def tracelog(tracker, path):
    import gc
    import tracemalloc
    from collections import Counter

    while True:
        time.sleep(1200)
        snap = tracemalloc.take_snapshot()
        rss = [l for l in open("/proc/self/status") if l.startswith("VmRSS")][
            0
        ].split()[1]
        cur, peak = tracemalloc.get_traced_memory()
        types = Counter(type(o).__name__ for o in gc.get_objects())
        with open(path, "a") as f:
            f.write(
                f"=== h={(time.time() - tracker.t0) / 3600:.2f} round={max(tracker.rows, default=-1)} rss={int(rss) / 2**20:.2f}G traced={cur / 2**30:.2f}G peak={peak / 2**30:.2f}G\n"
            )
            f.write("gc types: " + str(types.most_common(12)) + "\n")
            for st in snap.statistics("lineno")[:20]:
                f.write(
                    f"  {st.size / 2**20:9.1f}M {st.count:10d}  {st.traceback[0]}\n"
                )
            for st in snap.statistics("traceback")[:6]:
                f.write(
                    f" TB {st.size / 2**20:9.1f}M "
                    + " <- ".join(str(fr) for fr in reversed(st.traceback))
                    + "\n"
                )
        del snap


def memlog(pst, tracker, path):
    import orthogonal_dfa.l_star.counterexample_synthesis as cs

    while True:
        try:
            t = pst.table
            masks = list(t._masks)
            memo = getattr(t.memo, "_table", None)
            rss = [l for l in open("/proc/self/status") if l.startswith("VmRSS")][
                0
            ].split()[1]
            row = dict(
                h=round((time.time() - tracker.t0) / 3600, 3),
                rss_gb=int(rss) / 2**20,
                round=max(tracker.rows, default=-1),
                fnr_limit=getattr(pst, "fnr_limit", pst.config.fnr_limit),
                prefixes=t.num_prefixes,
                suffixes=len(t._suffixes),
                masks_gb=sum(m.nbytes for m in masks) / 2**30,
                pool=len(getattr(pst, "suffix_pool", ())),
                memo_entries=(
                    len(memo)
                    if memo is not None
                    else len(getattr(t.memo, "_cache", {}))
                ),
                memo_gb=(
                    (memo._lo.nbytes + memo._hi.nbytes + memo._values.nbytes) / 2**30
                    if memo is not None
                    else -1
                ),
                populations={str(k): len(v) for k, v in list(t._populations.items())},
            )
        except Exception as e:
            row = dict(error=repr(e))
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")
        time.sleep(300)


def log_batches(path):
    from orthogonal_dfa.superlanguage.oracle import LiftedOracle

    inner = LiftedOracle.membership_queries

    def wrapped(self, strings):
        out = inner(self, strings)
        if len(strings) >= 50_000:
            st = {
                l.split(":")[0]: int(l.split()[1]) / 2**20
                for l in open("/proc/self/status")
                if l.startswith(("VmRSS", "VmHWM"))
            }
            with open(path, "a") as f:
                f.write(
                    json.dumps(
                        dict(
                            t=time.time(),
                            n=len(strings),
                            rss_gb=st["VmRSS"],
                            hwm_gb=st["VmHWM"],
                        )
                    )
                    + "\n"
                )
        return out

    LiftedOracle.membership_queries = wrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--max-rounds", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-halving", action="store_true")
    ap.add_argument("--plateau", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log_batches(f"{a.out}/batches.jsonl")
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short=8", "HEAD"], text=True
    ).strip()
    base = gate_residual_oracle(
        default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85
    )
    oc = lambda _noise, s: LiftedOracle(base, VOCAB, seed=s)
    halves = held_out(oc(None, a.seed))
    pst = build_pst(
        oc, min_signal_strength=0.05, seed=a.seed, sampler=SuperSampler(VOCAB, 36)
    )
    if a.restarts > 1:
        pst.config.counterexample_restarts = a.restarts
    if a.plateau:
        pst.config.fnr_limit = 0.10
        pst.config.fnr_plateau_patience = 3
        pst.config.counterexample_score_patience = 500
        pst.config.counterexample_fnr_harvest = True
        pst.config.counterexample_restarts = a.restarts if a.restarts > 1 else 3
        pst.config.fnr_plateau_tol = 0.01
    tracker = Tracker(halves, a.out)
    if a.no_halving:
        from orthogonal_dfa.l_star import counterexample_synthesis as cs

        cs._blocked_at_limit = lambda *args: False
    if os.environ.get("TRACEMALLOC"):
        import tracemalloc

        threading.Thread(
            target=tracelog, args=(tracker, f"{a.out}/trace.txt"), daemon=True
        ).start()
    threading.Thread(
        target=memlog, args=(pst, tracker, f"{a.out}/mem.jsonl"), daemon=True
    ).start()
    best = counterexample_driven_synthesis(
        pst, acc_threshold=0.98, tracker=tracker, max_rounds=a.max_rounds
    )
    phi, top, used = score(best.dfa, halves)
    summary = dict(
        commit=commit,
        seed=a.seed,
        restarts=pst.config.counterexample_restarts,
        best_round=best.round_index,
        best_consistency=best.consistency,
        certified=getattr(best, "certified", None) is not None,
        best_phi=phi,
        best_states=len(best.dfa.states),
        best_top_state_share=top,
        rounds=len(tracker.rows),
        max_rounds=a.max_rounds,
        hours=(time.time() - tracker.t0) / 3600,
    )
    from orthogonal_dfa.l_star import counterexample_synthesis as cs

    summary["restarts_eval"] = [
        dict(
            round=r,
            restart=i,
            select_score=sel,
            phi=score(d, halves)[0],
            states=len(d.states),
        )
        for r, i, sel, d in getattr(cs, "RESTART_DFAS", [])
    ]
    with open(f"{a.out}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
