"""A family that leaves every state undecided a tenth of the time is one the FNR
gate admits, and it blocks the counterexample pass on a large enough tree: a
search needs dozens of reads decided at once.  Re-reading over a reserve is what
unblocks it."""

import contextlib
import io
import unittest

import numpy as np
from automata.fa.dfa import DFA

from orthogonal_dfa.l_star.examples.benchmark_generator import DFAOracle
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.lstar import estimate_agreement_rate
from orthogonal_dfa.l_star.structures import NoisyOracle
from orthogonal_dfa.l_star.suffix_family import REREAD_FAMILIES
from orthogonal_dfa.l_star.transition_resolver import TransitionResolver

MODULUS = 32
FAMILY = 62
#: Suffixes that flip every label, so each state reads undecided 8.8% of the time.
FLIPPED = 8


def _counter():
    return DFA(
        states=set(range(MODULUS)),
        input_symbols={0, 1},
        transitions={q: {0: q, 1: (q + 1) % MODULUS} for q in range(MODULUS)},
        initial_state=0,
        final_states=set(range(MODULUS // 2)),
        allow_partial=False,
    )


def _families(pst, rng, count):
    """``count`` families of ``FAMILY`` suffixes each: most keep the count of 1s
    mod ``MODULUS``, which preserves every label, and ``FLIPPED`` shift it by half,
    which flips every label."""
    drawn = set()

    def suffix_with_ones(r):
        while True:
            v = bytes(rng.integers(0, 2, size=2 * MODULUS).tolist())
            if sum(v) % MODULUS == r and v not in drawn:
                drawn.add(v)
                return pst.table.intern_suffix(v)

    families = []
    for _ in range(count):
        family = [suffix_with_ones(0) for _ in range(FAMILY - FLIPPED)]
        family += [suffix_with_ones(MODULUS // 2) for _ in range(FLIPPED)]
        rng.shuffle(family)
        families.append(family)
    return families


def _pass(rereading):
    target = _counter()
    creator = lambda nm, s: NoisyOracle(DFAOracle(target), nm, s)  # noqa: E731
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        pst = build_pst(creator, min_signal_strength=0.3, seed=0)
        vs, *reserve = _families(pst, np.random.default_rng(1000), 1 + REREAD_FAMILIES)
        pst.decision_boundary = 0.5
        pst.evidence_margin = 0.15
        chunks = list(reserve) if rereading else []
        resolver = TransitionResolver(pst, vs, lambda: chunks.pop(0) if chunks else [])
        resolver.close_edges()
        resolver.counterexample_pass(max_probes=4000, patience=149)
        dfa, tree = resolver.to_dfa_and_tree()
        agreement = estimate_agreement_rate(
            pst,
            pst.sampler,
            pst.oracle,
            tree,
            dfa,
            num_samples=2000,
            acc_threshold=0.98,
        )
    return len(resolver.indecisive), agreement


class TestRereadUnblocksThePass(unittest.TestCase):
    def test_the_reserve_settles_the_reads_that_blocked_the_pass(self):
        blocked, refused_at = _pass(rereading=False)
        settled, passed_at = _pass(rereading=True)

        self.assertLess(refused_at, 0.98)
        self.assertGreaterEqual(passed_at, 0.98)
        self.assertLess(10 * settled, blocked)


if __name__ == "__main__":
    unittest.main()
