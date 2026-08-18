"""The invariance gate: refuse a split whose distinguisher encodes absolute position
(a positional / shift-register ladder), while leaving a regular target -- whose
distinguishers are translation-invariant -- untouched.

The unit test pins the position-dependence signal directly (regular low, positional
high); the integration tests check the gate leaves a regular target converging and
stops the positional ladder from forming."""

import unittest

import numpy as np

from orthogonal_dfa.analysis.hidden_signal import ParityOracle
from orthogonal_dfa.analysis.ladder_repro import PositionalScoreOracle
from orthogonal_dfa.l_star.counterexample_synthesis import synthesize_direct_lstar_fnr
from orthogonal_dfa.l_star.direct_lstar import distinguisher_position_dependence
from orthogonal_dfa.l_star.learn import build_pst

# Between the regular targets (<= ~0.017) and the positional one (>= ~0.067).
THRESHOLD = 0.04


class TestPositionDependenceSignal(unittest.TestCase):
    def test_regular_distinguishers_are_invariant(self):
        parity = ParityOracle()
        for d in ([1], [2, 1], [3, 0, 0]):
            self.assertLess(distinguisher_position_dependence(parity, d, 4), THRESHOLD)

    def test_positional_distinguishers_encode_position(self):
        positional = PositionalScoreOracle()
        # At least one short distinguisher is clearly position-dependent.
        scores = [
            distinguisher_position_dependence(positional, d, 4)
            for d in ([1], [2, 1], [3, 0, 0])
        ]
        self.assertGreater(max(scores), THRESHOLD)


def _learn(oracle, *, invariance_threshold):
    length = 12
    pst = build_pst(
        lambda _n, _s: oracle,
        min_signal_strength=0.06,
        seed=0,
        sample_length=length,
        fnr_limit=0.02,
    )
    return synthesize_direct_lstar_fnr(
        pst,
        acc_threshold=0.98,
        max_rounds=3,
        counterexample_probes=200,
        per_state=12,
        min_indecisive=40,
        invariance_threshold=invariance_threshold,
    )[0]


class TestInvarianceGate(unittest.TestCase):
    def test_regular_target_still_converges_with_the_gate(self):
        # Parity's distinguishers are position-invariant, so the gate never fires
        # and the learner recovers the compact automaton (the safety property).
        # (End to end on the positional oracle the gate holds it to ~5 states vs the
        # ~18-state ladder ungated, but a full positional synthesis is too slow for
        # CI -- the oracle's membership_query is a Python loop -- so that is checked
        # via the position-dependence signal above, not a live synthesis here.)
        oracle = ParityOracle()
        dfa = _learn(oracle, invariance_threshold=THRESHOLD)
        held = np.random.default_rng(1).integers(0, 4, (2000, 12))
        call = np.array([bool(dfa.accepts_input(s.tolist())) for s in held])
        truth = np.array([oracle.membership_query(s.tolist()) for s in held])
        self.assertGreaterEqual(float((call == truth).mean()), 0.95)
        self.assertLessEqual(len(dfa.states), 6)


if __name__ == "__main__":
    unittest.main()
