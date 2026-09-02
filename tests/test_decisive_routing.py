"""Reproducer for the confident-band routing collapse and its decisive-routing fix.

The target is the frame-0-and-1 stop-codon rule over the k-mer + wildcard
superlanguage, plus a weak, content-correlated substructure -- *which* reading
frame closed first.  That bit genuinely distinguishes two states the coarse rule
merges (so it is a real refinement, not noise), but its accept-rate gap sits right
at the confident band.

With a confident-band sift (the old behaviour), the borderline successors of a
split cannot be placed: ``EdgeResolver.decisive_target`` finds no decisive target
and the export self-loops the edge, misrouting every string that passes through it
into the reject sink.  The run collapses -- to accept-all on some seeds, to a
tangled partial refinement on others -- and *which* happens is seed-fragile.

Routing decisively (a zero-width band, :meth:`SuffixFamily.side`) places every
successor, so the learner instead recovers the correct finer DFA on every seed.
Without the fix this test fails (the collapse seeds agree far less with the rule);
with it, every seed lands the finer DFA.
"""

import unittest

import numpy as np

from orthogonal_dfa.l_star.counterexample_synthesis import (
    counterexample_driven_synthesis,
)
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.structures import Oracle, _uniform_random
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

KMERS = ((3, 0, 2), (3, 0, 0), (3, 2, 0))  # TAG, TAA, TGA
SIGNAL = 0.30  # oracle accept/reject probability is 0.5 +/- SIGNAL
DELTA = 0.10  # substructure strength: a +/-DELTA accept-rate tilt by first frame
LENGTH = 36  # super-symbols per string
SEEDS = (0, 1, 2)


def _frame(symbols):
    """``(both_frames_closed, first_sign)`` for a super-string: ``first_sign`` is
    ``+1`` when a phase-0 k-mer precedes any phase-1 one, ``-1`` the other way,
    ``0`` when neither reading frame is closed.  Phase is the running wildcard
    count mod 3 (a wildcard is any symbol past the k-mers)."""
    wild = 0
    f0 = f1 = False
    first = 0
    for c in symbols:
        if c >= len(KMERS):
            wild += 1
        else:
            phase = wild % 3
            if phase == 0:
                if not f0 and not f1 and first == 0:
                    first = +1
                f0 = True
            elif phase == 1:
                if not f0 and not f1 and first == 0:
                    first = -1
                f1 = True
    return (f0 and f1), first


class _SubstructureOracle(Oracle):
    """The frame-0-and-1 rule (weak, dominant) plus a content-correlated
    which-frame-first tilt of size ``DELTA``, deterministic per string."""

    def __init__(self, seed):
        self._seed = seed

    @property
    def alphabet_size(self):
        return len(KMERS) + 2

    def membership_query(self, string):
        both, first = _frame(string)
        p = (0.5 - SIGNAL) if both else (0.5 + SIGNAL)
        p = min(max(p + DELTA * first, 0.02), 0.98)
        return _uniform_random(bytes(string), self._seed) < p


def _learn(seed):
    vocab = KmerVocabulary(kmers=KMERS, base_alphabet_size=4)
    pst = build_pst(
        lambda _noise, s: _SubstructureOracle(s),
        min_signal_strength=0.20,
        seed=seed,
        sampler=SuperSampler(vocab, LENGTH),
    )
    pst.config.fnr_limit = 0.10
    for dfa, _dt, _acc, _boundary, _cls in counterexample_driven_synthesis(
        pst, acc_threshold=0.88
    ):
        return dfa, vocab
    raise AssertionError("synthesis produced no DFA")


class TestDecisiveRouting(unittest.TestCase):
    def test_borderline_substructure_recovers_the_finer_dfa(self):
        for seed in SEEDS:
            dfa, vocab = _learn(seed)
            rng = np.random.default_rng(seed + 999)
            samp = SuperSampler(vocab, LENGTH)
            ev = [list(samp.sample(rng, vocab.alphabet_size)) for _ in range(2000)]
            call = np.array([bool(dfa.accepts_input(w)) for w in ev])
            rule = np.array(
                [not _frame(w)[0] for w in ev]
            )  # accept iff not both closed
            agree = float(np.mean(call == rule))
            accept_rate = float(call.mean())
            # The collapse either accepts everything or tangles the routing; both
            # drop agreement with the rule well below the finer DFA's ~0.96.
            self.assertGreater(
                agree,
                0.88,
                f"seed {seed}: routing collapsed (agreement {agree:.3f}, "
                f"accept-rate {accept_rate:.3f}); expected the finer DFA",
            )
            self.assertLess(
                accept_rate,
                0.95,
                f"seed {seed}: collapsed to accept-all (accept-rate {accept_rate:.3f})",
            )


if __name__ == "__main__":
    unittest.main()
