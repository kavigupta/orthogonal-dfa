"""Probe whether the prepend-ladder *hides* recoverable signal.

See ``results/hidden_signal_probe.md`` for the writeup.  The short version: it does
not.  For genuinely-regular signal the learner recovers it up to the (distractor-
reduced) ceiling, so the ``PositionalScoreOracle`` "recovers nothing" in
``ladder_repro.py`` is *absence* of compact structure, not concealment of it.

Two experiments:

* ``HiddenFrameOracle`` -- a regular frame signal XOR-entangled with a rare positional
  flip.  The signal is real (a compact stop-codon DFA scores well above chance) but the
  flip is non-regular and cannot be routed away as noise.  direct-L* still recovers the
  frame signal at the ceiling.

* ``closure`` / ``cyclic_core`` -- measurable regularity signals.  A finite automaton
  over long strings must cycle (pigeonhole); a prepend-ladder grows forward forever.
  Both separate regular targets (parity, mod-k) from the positional ladder.

Run::

    python -m orthogonal_dfa.analysis.hidden_signal
"""

from __future__ import annotations

import contextlib
import io
import re
import sys

import numpy as np
from automata.fa.dfa import DFA

from orthogonal_dfa.analysis.ladder_repro import PositionalScoreOracle, nframes
from orthogonal_dfa.l_star.counterexample_synthesis import synthesize_direct_lstar_fnr
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.structures import Oracle


class HiddenFrameOracle(Oracle):
    """``[nframes(seq) >= thr]  XOR  [positional_score(seq) > tau]``.

    The frame term is the regular, ~50/50 stop-codon signal; the positional term is a
    rare (fraction ``flip``), non-regular perturbation, entangled by XOR so it corrupts
    the same strings the frame signal lives on.  ``tau`` is calibrated at construction
    to fire on a ``flip`` fraction of random strings of the given length.
    """

    alphabet_size = 4

    def __init__(self, *, length, thr=1, flip=0.2, seed=42, n_max=400, calib=8000):
        self.thr = thr
        rng = np.random.default_rng(seed)
        w = rng.normal(size=(n_max, 4))
        self._w = w - w.mean(axis=1, keepdims=True)
        scores = np.array([self._score(s) for s in rng.integers(0, 4, (calib, length))])
        self._tau = float(np.quantile(scores, 1 - flip))

    def _score(self, seq):
        return sum(self._w[i, seq[i]] for i in range(len(seq)))

    def frame(self, seq):
        return nframes(list(seq)) >= self.thr

    def membership_query(self, seq):
        seq = list(seq)
        return bool(self.frame(seq) ^ (self._score(seq) > self._tau))


class ParityOracle(Oracle):
    """Regular, 2-state cycle: parity of the count of symbol 0 (~50/50)."""

    alphabet_size = 4

    def membership_query(self, seq):
        return sum(1 for x in seq if x == 0) % 2 == 0


class ModOracle(Oracle):
    """Regular, k-state cycle: count of symbol 0 mod k == 0."""

    alphabet_size = 4

    def __init__(self, k=3):
        self.k = k

    def membership_query(self, seq):
        return sum(1 for x in seq if x == 0) % self.k == 0


def synthesize_capturing(oracle, *, length, rounds, seed=0):
    """Run synthesis, returning ``(dfa, unresolved_edge_count)``.  The unresolved count
    is the number of edges the resolver could not close and fell back to a self-loop --
    read off the warnings synthesis prints, which are otherwise swallowed here."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pst = build_pst(
            lambda _n, _s: oracle,
            min_signal_strength=0.06,
            seed=seed,
            sample_length=length,
            fnr_limit=0.02,
        )
        dfa, _ = synthesize_direct_lstar_fnr(
            pst,
            acc_threshold=0.98,
            max_rounds=rounds,
            counterexample_probes=150,
            per_state=12,
            min_indecisive=40,
        )
    return dfa, len(re.findall("no decisive edge", buf.getvalue()))


def cyclic_core(dfa):
    """States in an SCC of size >= 2 (self-loops excluded), plus the initial state."""
    adj = {s: set() for s in dfa.states}
    for s, row in dfa.transitions.items():
        for t in row.values():
            if t != s:
                adj[s].add(t)
    index, low, onstack, stack, counter, core = {}, {}, {}, [], [0], set()

    def strong(v):
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        onstack[v] = True
        for w in adj[v]:
            if w not in index:
                strong(w)
                low[v] = min(low[v], low[w])
            elif onstack.get(w):
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                onstack[w] = False
                comp.append(w)
                if w == v:
                    break
            if len(comp) >= 2:
                core.update(comp)

    sys.setrecursionlimit(100000)
    for v in dfa.states:
        if v not in index:
            strong(v)
    core.add(dfa.initial_state)
    return core


def core_extract(dfa, alphabet_size):
    """Collapse the acyclic, non-merging tail into the cyclic core.

    Keep the core states; redirect any core->tail edge to an absorbing sink labelled
    by the tail state's accept status -- i.e. at the point the ladder would start,
    commit to the classification instead of unrolling.  Returns ``(core_dfa, n_core)``.
    """
    core = cyclic_core(dfa)
    acc = set(dfa.final_states)
    sink_a, sink_r = "sinkA", "sinkR"
    trans = {}
    for s in core:
        trans[s] = {
            c: (
                t
                if (t := dfa.transitions[s][c]) in core
                else (sink_a if t in acc else sink_r)
            )
            for c in range(alphabet_size)
        }
    trans[sink_a] = {c: sink_a for c in range(alphabet_size)}
    trans[sink_r] = {c: sink_r for c in range(alphabet_size)}
    core_dfa = DFA(
        states=set(core) | {sink_a, sink_r},
        input_symbols=set(range(alphabet_size)),
        transitions=trans,
        initial_state=dfa.initial_state,
        final_states=(core & acc) | {sink_a},
        allow_partial=False,
    )
    return core_dfa, len(core)


def _accuracy(dfa, oracle, held):
    call = np.array([bool(dfa.accepts_input(s.tolist())) for s in held])
    ora = np.array([oracle.membership_query(s.tolist()) for s in held])
    return float((call == ora).mean()), float(ora.mean())


def frame_recovery(flip, *, length=16, seed=0):
    """Does the learner recover the frame signal buried in the positional flip?"""
    oracle = HiddenFrameOracle(length=length, flip=flip, seed=42)
    dfa, _ = synthesize_capturing(oracle, length=length, rounds=1, seed=seed)
    held = np.random.default_rng(seed + 1).integers(0, 4, (4000, length))
    acc, base = _accuracy(dfa, oracle, held)
    frame = np.array([oracle.frame(s.tolist()) for s in held])
    ora = np.array([oracle.membership_query(s.tolist()) for s in held])
    ceiling = float((frame == ora).mean())
    core_dfa, n_core = core_extract(dfa, oracle.alphabet_size)
    core_acc, _ = _accuracy(core_dfa, oracle, held)
    return {
        "flip": flip,
        "states": len(dfa.states),
        "acc": acc,
        "chance": max(base, 1 - base),
        "ceiling": ceiling,
        "core_states": n_core,
        "core_acc": core_acc,
    }


def regularity_row(name, oracle, *, length, rounds=2, seed=0):
    dfa, unresolved = synthesize_capturing(
        oracle, length=length, rounds=rounds, seed=seed
    )
    states = len(dfa.states)
    closure = 1 - unresolved / max(states * oracle.alphabet_size, 1)
    return {
        "name": name,
        "states": states,
        "closure": closure,
        "cyclic": len(cyclic_core(dfa)) / max(states, 1),
    }


def main():
    print("# Can the ladder hide recoverable signal?\n")
    print("## frame signal buried in a positional flip (recovered = at the ceiling)")
    print(
        f"{'flip':>5} {'states':>7} {'acc':>6} {'ceiling':>8} {'chance':>7} "
        f"{'core':>5} {'core_acc':>9}"
    )
    for flip in (0.2, 0.35):
        r = frame_recovery(flip)
        print(
            f"{r['flip']:>5.2f} {r['states']:>7} {r['acc']:>6.3f} "
            f"{r['ceiling']:>8.3f} {r['chance']:>7.3f} "
            f"{r['core_states']:>5} {r['core_acc']:>9.3f}"
        )

    print(
        "\n## closure / cyclic-core: regular targets close and merge, the ladder does not"
    )
    print(f"{'target':>14} {'states':>7} {'closure':>8} {'cyclic':>7}")
    rows = [
        regularity_row("parity", ParityOracle(), length=12),
        regularity_row("mod3", ModOracle(3), length=12),
        regularity_row("positional", PositionalScoreOracle(), length=12),
    ]
    for r in rows:
        print(
            f"{r['name']:>14} {r['states']:>7} {r['closure']:>8.2f} {r['cyclic']:>7.2f}"
        )


if __name__ == "__main__":
    main()
