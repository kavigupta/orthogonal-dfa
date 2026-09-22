"""Per-state vote margins at judge time: how close does any state get to a threshold?

For each judged family, group the representative prefixes by the state they reach in
the target DFA and record that state's vote mean, its distance to the nearer
threshold, and how its decisive prefixes split.  A state whose vote sits inside a
band-width of a threshold is one the proof's worst case is about.
"""
import contextlib
import io
import json
import os
import sys

root, task, seed, fpr, fnr, out = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6],
)
sys.path.insert(0, root)
src = open(os.path.join(root, "scripts", "count_queries.py")).read()
ns = {"__name__": "cq", "__file__": "count_queries.py"}
exec(compile(src, "count_queries.py", "exec"), ns)

import numpy as np
from automata.fa.dfa import DFA

import orthogonal_dfa.l_star.cluster as cluster
from orthogonal_dfa.l_star.examples.benchmark_generator import DFAOracle
from orthogonal_dfa.l_star.examples.bernoulli_parity import BernoulliParityOracle, BernoulliRegex
from orthogonal_dfa.l_star.learn import learn_dfa
from orthogonal_dfa.l_star.sampler import Sampler
from orthogonal_dfa.l_star.structures import AsymmetricBernoulli, NoisyOracle

MOTIF = bytes([1, 0, 1, 0, 1, 0, 1])


class NearMissSampler(Sampler):
    """Uniform strings, except ``share`` of them end one symbol short of MOTIF."""

    def __init__(self, length=40, share=0.3):
        self.length, self.share = length, share

    def symbol_weights(self, alphabet_size):
        return [1] * alphabet_size

    def sample(self, rng, alphabet_size):
        def draw(n):
            return rng.integers(0, alphabet_size, size=n, dtype=np.uint8).tobytes()

        if rng.random() >= self.share:
            return draw(self.length)
        for _ in range(50):
            candidate = draw(self.length - len(MOTIF) + 1) + MOTIF[:-1]
            if MOTIF not in candidate:
                return candidate
        return draw(self.length)


def build(task):
    """(oracle_creator, signal, noise, sampler) for the named task."""
    if task == "near_miss":
        return (
            lambda nm, s: NoisyOracle(BernoulliRegex(regex=r".*1010101.*"), nm, s),
            0.2, AsymmetricBernoulli(p_0=0.15, p_1=0.7), NearMissSampler(),
        )
    bench = ns["BENCHMARKS"][task]
    spec = bench["oracle"]
    if spec["kind"] == "modulo":
        inner = BernoulliParityOracle(modulo=spec["modulo"], allowed_moduluses=tuple(spec["allowed"]))
    elif spec["kind"] == "regex":
        inner = BernoulliRegex(regex=spec["regex"])
    else:
        P = ns["POOR_CASE_DFA"]
        inner = DFAOracle(DFA(states=set(range(10)), input_symbols={0, 1},
                              transitions={int(q): {int(a): int(r) for a, r in row.items()}
                                           for q, row in P["transitions"].items()},
                              initial_state=P["initial"], final_states=set(P["final"]),
                              allow_partial=False))
    noise = AsymmetricBernoulli(**bench["noise"]) if bench["noise"] else None
    return (lambda nm, s, _i=inner: NoisyOracle(_i, nm, s)), bench["signal"], noise, None


creator, signal, noise, sampler = build(task)
TARGET = creator(None, 0).inner.target_dfa()
ROUNDS = []
orig_judge = cluster.judge_family


def state_of(prefix):
    q = TARGET.initial_state
    for c in prefix:
        q = TARGET.transitions[q][c]
    return q


def judge(pst, gate, v, vs, family_size, *a, **k):
    j = orig_judge(pst, gate, v, vs, family_size, *a, **k)
    if j.reason != "undersized":
        rep = pst.table.representative
        prefixes = [p for p, keep in zip(pst.table.prefixes, rep) if keep]
        decision = pst.compute_decision(j.vs, rep)
        states = np.array([state_of(p) for p in prefixes])
        acc, rej = pst.accept_thresh, pst.reject_thresh
        per = {}
        for q in sorted(set(states.tolist())):
            d = decision[states == q]
            decisive = (d >= acc) | (d < rej)
            accepted = int((d >= acc).sum())
            per[str(q)] = dict(
                n=int(len(d)), vote=round(float(d.mean()), 4),
                margin=round(float(min(abs(d.mean() - acc), abs(d.mean() - rej))), 4),
                truth=int(q in TARGET.final_states),
                decisive=round(float(decisive.mean()), 3),
                minority=int(min(accepted, int(decisive.sum()) - accepted)),
            )
        ROUNDS.append(dict(k=len(j.vs), m=len(prefixes), fnr=round(float(j.fnr), 4),
                           accept=round(float(acc), 4), reject=round(float(rej), 4),
                           states=per))
        json.dump(dict(task=task, seed=seed, rates=[fpr, fnr], rounds=ROUNDS), open(out, "w"))
    return j


cluster.judge_family = judge
kw = dict(noise_model=noise) if noise else {}
if sampler is not None:
    kw["sampler"] = sampler
status = "ok"
try:
    with contextlib.redirect_stdout(io.StringIO()):
        learn_dfa(creator, min_signal_strength=signal, seed=seed, **kw)
except Exception as e:
    status = f"{type(e).__name__}: {str(e)[:150]}"
json.dump(dict(task=task, seed=seed, rates=[fpr, fnr], status=status, rounds=ROUNDS), open(out, "w"))
