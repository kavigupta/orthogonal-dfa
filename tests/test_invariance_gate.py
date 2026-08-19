"""The invariance gate: refuse a split whose distinguisher encodes absolute position
(a positional / shift-register ladder), while leaving a regular target -- whose
distinguishers are translation-invariant -- untouched.

The pathology is a target the learner cannot represent as a compact DFA because its
decision depends on absolute position (a positional score), so the discrimination
tree grows an unbounded ladder of ever-longer distinguishers. A genuine finite-memory
feature is transportable -- the same feature wherever it occurs -- so appending its
distinguisher has a position-invariant effect; a positional feature does not. The gate
measures exactly that per distinguisher and refuses the position-encoding ones.

Unit tests pin the signal (a regular distinguisher's Bayes factor favours invariance,
a positional one favours position-dependence); the integration test checks the gate
leaves a regular target converging (the safety property -- the gate is otherwise dead
code, active only when ``invariance_gate=True``)."""

import unittest

import numpy as np

from orthogonal_dfa.l_star.counterexample_synthesis import (
    do_counterexample_driven_synthesis,
)
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.structures import Oracle
from orthogonal_dfa.l_star.transition_resolver import (
    distinguisher_position_log_bayes_factor,
)


class ParityOracle(Oracle):
    """Regular, 2-state cycle: parity of the count of symbol 0 (~50/50)."""

    alphabet_size = 4

    def membership_query(self, string):
        return sum(1 for x in string if x == 0) % 2 == 0


class PositionalScoreOracle(Oracle):
    """Non-regular: accept iff ``sum_i W[i, string[i]] > 0`` for a fixed random,
    position-specific, per-position-centered weight table (so ~50/50 balanced).  The
    score is positional, so there is no compact automaton to close on."""

    alphabet_size = 4

    def __init__(self, *, n_max=400, seed=42):
        w = np.random.default_rng(seed).normal(size=(n_max, 4))
        self._w = w - w.mean(axis=1, keepdims=True)

    def membership_query(self, string):
        seq = list(string)
        return bool(sum(self._w[i, seq[i]] for i in range(len(seq))) > 0.0)


class LatePositionalOracle(Oracle):
    """Positional score over only positions >= ``start`` -- position-dependence that
    lives beyond a short fixed probe window, so it is caught only when the window
    reaches the operating length."""

    alphabet_size = 4

    def __init__(self, *, start, n_max=400, seed=3):
        w = np.random.default_rng(seed).normal(size=(n_max, 4))
        self._w = w - w.mean(axis=1, keepdims=True)
        self._start = start

    def membership_query(self, string):
        seq = list(string)
        return bool(sum(self._w[i, seq[i]] for i in range(self._start, len(seq))) > 0.0)


class InteractionOracle(Oracle):
    """Non-regular, and harder than PositionalScoreOracle: a dense positional score
    plus prefix-boundary interaction terms (a random ``W2[.,.]`` for pairs of positions
    near ``length``), median-thresholded.  Its distinguishers are only *mildly*
    position-dependent per length -- enough that a fixed-threshold position score misses
    them (the old residual-std gate scored them ~0.038, under its 0.04 threshold) -- but
    the Bayes factor, pooling evidence across probe lengths, still refuses them."""

    alphabet_size = 4

    def __init__(self, *, length=16, n_pairs=32, seed=7):
        rng = np.random.default_rng(seed)
        self._m = 4 * length
        w1 = rng.normal(size=(self._m, 4))
        self._w1 = w1 - w1.mean(1, keepdims=True)
        self._pairs = [
            (
                int(rng.integers(0, length)),
                int(rng.integers(length - 4, length + 4)),
                rng.normal(size=(4, 4)),
            )
            for _ in range(n_pairs)
        ]
        sample = rng.integers(0, 4, (4000, 2 * length))
        self._thr = float(np.median([self._score(r.tolist()) for r in sample]))

    def _score(self, seq):
        n = min(len(seq), self._m)
        score = sum(self._w1[i, seq[i]] for i in range(n))
        for i, j, w in self._pairs:
            if i < n and j < n:
                score += 4.0 * w[seq[i], seq[j]]
        return score

    def membership_query(self, string):
        return bool(self._score(list(string)) > self._thr)


class TestPositionDependenceSignal(unittest.TestCase):
    def test_regular_distinguishers_favour_invariance(self):
        parity = ParityOracle()
        for d in ([1], [2, 1], [3, 0, 0]):
            self.assertLess(
                distinguisher_position_log_bayes_factor(parity, d, 4, sample_length=16),
                0.0,
            )

    def test_positional_distinguishers_favour_position_dependence(self):
        positional = PositionalScoreOracle()
        scores = [
            distinguisher_position_log_bayes_factor(positional, d, 4, sample_length=16)
            for d in ([1], [2, 1], [3, 0, 0])
        ]
        self.assertGreater(max(scores), 0.0)

    def test_probe_window_tracks_the_operating_length(self):
        # Position-dependence living beyond a short fixed window (positions >= 28) is
        # caught when the learner operates where it lives (length 40), and correctly
        # not flagged where those positions do not exist (length 24) -- so the window
        # must follow sample_length, not a hardcoded range.
        late = LatePositionalOracle(start=28)
        ds = ([1], [2, 1], [1, 1, 1])
        at_40 = max(
            distinguisher_position_log_bayes_factor(late, d, 4, sample_length=40)
            for d in ds
        )
        at_24 = max(
            distinguisher_position_log_bayes_factor(late, d, 4, sample_length=24)
            for d in ds
        )
        self.assertGreater(at_40, 0.0)
        self.assertLess(at_24, 0.0)

    def test_interaction_distinguishers_favour_position_dependence(self):
        # The interaction oracle is the case the old fixed-threshold gate missed: its
        # C/G-run distinguishers (1=C, 2=G) are only mildly position-dependent per
        # length, so a fixed threshold under-measures them -- but the Bayes factor,
        # accumulating evidence across probe lengths, refuses them.
        oracle = InteractionOracle()
        for d in ([1, 1], [2, 1], [1, 1, 1]):
            self.assertGreater(
                distinguisher_position_log_bayes_factor(oracle, d, 4, sample_length=16),
                0.0,
            )


class TestInvarianceGate(unittest.TestCase):
    def test_regular_target_still_converges_with_the_gate(self):
        # Parity's distinguishers are position-invariant, so the gate never fires
        # and the learner recovers the compact automaton (the safety property).
        oracle = ParityOracle()
        pst = build_pst(
            lambda _n, _s: oracle,
            min_signal_strength=0.06,
            seed=0,
            sample_length=12,
        )
        dfa, _ = do_counterexample_driven_synthesis(
            pst, acc_threshold=0.98, invariance_gate=True
        )
        held = np.random.default_rng(1).integers(0, 4, (2000, 12))
        call = np.array([bool(dfa.accepts_input(s.tolist())) for s in held])
        truth = np.array([oracle.membership_query(s.tolist()) for s in held])
        self.assertGreaterEqual(float((call == truth).mean()), 0.95)
        self.assertLessEqual(len(dfa.states), 6)


if __name__ == "__main__":
    unittest.main()
