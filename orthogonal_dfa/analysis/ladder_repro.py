"""Reproduce the direct-L* prepend-*ladder*, and separate its two causes.

Phenomenon
----------
On the composition-deconfounded SpliceAI oracle, direct-L* does not converge to a
compact DFA.  The discrimination tree grows a *ladder* of distinguishers, each one a
single symbol prepended to a previous one::

    CAAGCG -> ACAAGCG -> AACAAGCG -> CAACAAGCG -> TCAACAAGCG -> GTCAACAAGCG

and the exported DFA recovers *no* signal (held-out accuracy at chance).  This module
reproduces that with ~20-line programmatic oracles and shows the ladder has two
independent ingredients that must be told apart.

Why the ladder forms
--------------------
**1. New distinguishers can only grow by prepending.**  When a counterexample exposes
an ambiguous edge -- state ``s1`` --``c``--> two different leaves -- the learner builds
the separating distinguisher with ``first_disagreement(witness, sprime, prefix=[c])``
(``direct_lstar._act_on_disagreement``), which returns ``(*[c], *node_midfix)``: the
edge symbol prepended to an *existing* tree distinguisher
(``MidfixTree.first_disagreement``).  So every new distinguisher is one symbol longer
than one already present -- distinguishers can only ever grow, on the left.

**2. The prepend rule ladders forever only when the DFA can never agree with the
oracle.**  A target with genuine compact finite-state structure reuses short
distinguishers and *closes* -- the counterexamples run out.  The chain grows without end
only when the learner can never make the DFA agree with the oracle, so the
counterexample search never runs dry.

These are different things, and the two oracles below separate them:

* ``FrameOracle`` -- ``nframes(s) >= thr`` (reading-frame closure) diluted with balanced
  hash noise.  This is **regular**: its minimal DFA is a chain tracking, per frame,
  "have I seen a stop yet".  direct-L* ladders on it (the chain *is* a stop-codon
  prepend-ladder) but **converges** and recovers the clean signal perfectly at every
  noise level.  This is the *control*: the ladder alone is not the pathology.

* ``PositionalScoreOracle`` -- ``sum_i W[i, s[i]] > 0`` for a fixed random,
  position-specific, per-position-centered weight table (so ~50/50 balanced).  This is
  **positional, non-compositional, and not compactly regular** -- structurally what
  SpliceAI's residual is.  direct-L* ladders on it, adds states without bound, and
  recovers *nothing* (held-out accuracy at chance).  This is the *pathology*.

The discriminator between "ladders and converges" and "ladders forever, recovers
nothing" is exactly whether the target is compactly regular in the region the midfix
varies -- which is why the composition-deconfounded SpliceAI oracle (positional,
non-regular) shatters where the frame automaton (regular) is learned fine.

Run
---
    python -m orthogonal_dfa.analysis.ladder_repro
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

from orthogonal_dfa.l_star.counterexample_synthesis import synthesize_direct_lstar_fnr
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.structures import Oracle

# TAA, TGA, TAG as tuples over A=0, C=1, G=2, T=3.
_STOPS = {(3, 0, 0), (3, 2, 0), (3, 0, 2)}


def nframes(seq):
    """How many of the 3 reading frames contain a stop codon (0..3)."""
    n = 0
    for phase in range(3):
        sub = seq[phase:]
        if any(tuple(sub[i : i + 3]) in _STOPS for i in range(0, len(sub) - 2, 3)):
            n += 1
    return n


def _hash01(seq, salt):
    digest = hashlib.blake2b(bytes([salt]) + bytes(seq), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


class FrameOracle(Oracle):
    """REGULAR control.  accept iff nframes(seq) >= thr, with a fraction ``1 - sw`` of
    strings routed to a balanced deterministic coin instead (structureless noise).

    ``nframes >= thr`` is a regular language whose minimal DFA is a chain, and it is
    ~50/50 balanced.  direct-L* ladders on the stop-codon chain but converges.
    """

    alphabet_size = 4

    def __init__(self, *, sw: float = 0.55, thr: int = 2):
        self.sw = sw
        self.thr = thr

    def membership_query(self, string):
        seq = list(string)
        if _hash01(seq, 1) < self.sw:
            return nframes(seq) >= self.thr
        return _hash01(seq, 2) < 0.5


class PositionalScoreOracle(Oracle):
    """NON-REGULAR pathology.  accept iff ``sum_i W[i, seq[i]] > 0`` for a fixed random,
    position-specific weight table centered per position (so ~50/50 balanced).

    The score is positional (not a function of composition) and continuous, so there is
    no compact automaton to close on -- structurally what SpliceAI's residual is.  This
    is where the ladder never terminates and the DFA recovers nothing.
    """

    alphabet_size = 4

    def __init__(self, *, n_max: int = 400, seed: int = 42):
        w = np.random.default_rng(seed).normal(size=(n_max, 4))
        self._w = w - w.mean(axis=1, keepdims=True)  # center => E[score]=0 => ~50%

    def membership_query(self, string):
        seq = list(string)
        return bool(sum(self._w[i, seq[i]] for i in range(len(seq))) > 0.0)


class InteractionOracle(Oracle):
    """A HARDER non-regular pathology than :class:`PositionalScoreOracle`.

    A dominant dense linear score (which spreads prefix propensities, so the FNR
    machinery still builds a family and splits) PLUS prefix-boundary interaction terms:
    the effect of the boundary window where a midfix is inserted depends on the prefix,
    so a midfix distinguisher separates prefixes differently -- the driver of the prepend
    ladder, absent from the pure halfspace.  It shatters where ``PositionalScoreOracle``
    only mildly ladders (scaling with ``alpha``).

    It is also the case the **translation-invariance split gate does not catch**: its
    distinguishers are C/G runs that only weakly encode absolute position
    (``distinguisher_position_dependence`` ~0.038, below the ~0.04 threshold the
    shift-register ladder's ~0.084 exceeds), so the gate stays inert and the oracle still
    shatters.  Median-thresholded (~50/50 balanced).
    """

    alphabet_size = 4

    def __init__(
        self,
        *,
        length: int = 40,
        n_pairs: int = 120,
        alpha: float = 4.0,
        boundary_window: int = 8,
        seed: int = 7,
    ):
        self._alpha = alpha
        self._max_len = 6 * length
        qlen = 2 * length
        rng = np.random.default_rng(seed)
        w1 = rng.normal(size=(self._max_len, 4))
        self._w1 = w1 - w1.mean(axis=1, keepdims=True)  # dense linear (spreads propensity)
        # interaction pairs: prefix position i x boundary position j near `length`
        self._pairs = []
        for _ in range(n_pairs):
            i = int(rng.integers(0, length))
            j = int(
                rng.integers(
                    max(0, length - boundary_window), min(qlen, length + boundary_window)
                )
            )
            self._pairs.append((i, j, rng.normal(size=(4, 4))))
        sample = rng.integers(0, 4, (6000, qlen))
        self._threshold = float(
            np.median([self._score(row.tolist()) for row in sample])
        )

    def _score(self, seq):
        n = min(len(seq), self._max_len)
        score = sum(self._w1[i, seq[i]] for i in range(n))
        for i, j, w in self._pairs:
            if i < n and j < n:
                score += self._alpha * w[seq[i], seq[j]]
        return score

    def membership_query(self, string):
        return bool(self._score(list(string)) > self._threshold)

    def membership_queries(self, strings):
        """Vectorised batch of :meth:`membership_query` (numerically identical, ~10x
        faster) -- the synthesiser and the position-dependence probe both query in
        bulk, and the per-string Python ``_score`` loop is what makes the repro slow."""
        rows = [list(s) for s in strings]
        if not rows:
            return np.zeros(0, dtype=bool)
        lengths = np.array([len(s) for s in rows])
        pair_span = max((max(i, j) for i, j, _ in self._pairs), default=-1) + 1
        width = max(int(lengths.max()), pair_span)
        packed = np.zeros((len(rows), width), dtype=np.int64)
        for k, s in enumerate(rows):
            packed[k, : len(s)] = s
        used = np.minimum(lengths, self._max_len)  # positions _score sums over
        cols = np.arange(width)
        in_range = cols[None, :] < used[:, None]
        # masked-out columns contribute 0, so clip the w1 row index into bounds.
        linear = np.where(
            in_range, self._w1[np.minimum(cols, self._max_len - 1)[None, :], packed], 0.0
        ).sum(1)
        inter = np.zeros(len(rows))
        for i, j, w in self._pairs:
            valid = (i < used) & (j < used)
            inter += np.where(valid, self._alpha * w[packed[:, i], packed[:, j]], 0.0)
        return (linear + inter) > self._threshold


def distinguishers(tree):
    """Every internal-node midfix of the discrimination tree, root (ε) excluded."""
    out = []

    def walk(node):
        if isinstance(node, int):
            return
        midfix, lookup = node
        if len(midfix):
            out.append(tuple(midfix))
        walk(lookup[True])
        walk(lookup[False])

    walk(tree.root)
    return out


def prepend_chains(dset):
    """Decompose distinguishers into maximal prepend-chains ``d, d[1:], d[2:], ...``
    where every step drops the first symbol and lands on another distinguisher.  A
    chain of length >= 3 is a ladder."""
    present = set(dset)
    tails = {d[1:] for d in present if len(d) >= 1}
    heads = [d for d in present if d not in tails]
    chains = []
    for head in sorted(heads, key=len, reverse=True):
        chain, cur = [head], head
        while cur[1:] in present:
            cur = cur[1:]
            chain.append(cur)
        chains.append(chain)
    return chains


def _s(t):
    return "".join("ACGT"[c] for c in t) or "ε"


def run(oracle, *, length, seed=0, max_rounds=3):
    """Run direct-L* against ``oracle`` and summarise the discrimination tree."""
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, 4, (4000, length))
    base_rate = float(np.mean([oracle.membership_query(s.tolist()) for s in sample]))

    pst = build_pst(
        lambda _noise, _seed: oracle,
        min_signal_strength=0.06,
        seed=seed,
        sample_length=length,
        fnr_limit=0.02,
    )
    dfa, tree = synthesize_direct_lstar_fnr(
        pst,
        acc_threshold=0.98,
        max_rounds=max_rounds,
        counterexample_probes=400,
        per_state=20,
        min_indecisive=80,
    )

    dset = distinguishers(tree)
    chains = prepend_chains(dset)
    held = rng.integers(0, 4, (3000, length))
    dfa_call = np.array([bool(dfa.accepts_input(s.tolist())) for s in held])
    ora_call = np.array([oracle.membership_query(s.tolist()) for s in held])
    # accuracy vs the majority-class baseline: chance = max(p, 1-p)
    chance = max(base_rate, 1 - base_rate)
    return {
        "base_rate": base_rate,
        "states": len(dfa.states),
        "n_distinguishers": len(dset),
        "max_len": max((len(d) for d in dset), default=0),
        "longest_chain": max((len(c) for c in chains), default=0),
        "held_out_acc": float((dfa_call == ora_call).mean()),
        "chance": chance,
        "chains": chains,
    }


def _report(title, res):
    print(f"===== {title} " + "=" * (60 - len(title)))
    print(
        f"base accept-rate {res['base_rate']:.3f} | states {res['states']} | "
        f"distinguishers {res['n_distinguishers']} | max len {res['max_len']} | "
        f"longest prepend-chain {res['longest_chain']}"
    )
    print(
        f"held-out acc vs oracle {res['held_out_acc']:.3f}  "
        f"(chance = majority class = {res['chance']:.3f})"
    )
    for chain in res["chains"]:
        if len(chain) >= 3:
            print("   ladder: " + " <- ".join(_s(d) for d in chain))
    print()


def main():
    length = int(os.environ.get("LEN", "48"))
    print(f"# direct-L* prepend-ladder reproduction  (sample length {length})\n")

    control = run(FrameOracle(sw=0.55, thr=2), length=length)
    _report("CONTROL: FrameOracle (regular) -- ladders but CONVERGES", control)

    pathology = run(PositionalScoreOracle(), length=length)
    _report(
        "PATHOLOGY: PositionalScoreOracle (non-regular) -- ladders FOREVER", pathology
    )

    print("## the contrast")
    print(
        f"{'oracle':>28} {'regular?':>9} {'states':>7} {'chain':>6} "
        f"{'acc':>6} {'chance':>7} {'recovers?':>10}"
    )
    for name, reg, r in [
        ("FrameOracle", "yes", control),
        ("PositionalScoreOracle", "no", pathology),
    ]:
        recovers = "yes" if r["held_out_acc"] > r["chance"] + 0.03 else "NO"
        print(
            f"{name:>28} {reg:>9} {r['states']:>7} {r['longest_chain']:>6} "
            f"{r['held_out_acc']:>6.3f} {r['chance']:>7.3f} {recovers:>10}"
        )
    print("\nSame prepend-ladder mechanism; opposite outcome.  The discriminator is")
    print("whether the target is compactly regular in the region the midfix varies.")


if __name__ == "__main__":
    main()
