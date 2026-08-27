import contextlib
import io
import unittest
import unittest.mock

import numpy as np
from automata.fa.dfa import DFA

from orthogonal_dfa.l_star.lstar import denoise_accept_labels
from orthogonal_dfa.l_star.statistics import (
    binomial_side_of_boundary,
    denoise_sample_size,
)
from tests.dfas import PARITY, parity_membership

#: One state reachable by a single string of any length, the other by the rest.
ALL_ONES = DFA(
    states={0, 1},
    input_symbols={0, 1},
    transitions={0: {0: 1, 1: 0}, 1: {0: 1, 1: 1}},
    initial_state=0,
    final_states={0},
    allow_partial=False,
)


class _CountingOracle:
    """Noiseless parity oracle that records the size of every call it receives."""

    def __init__(self):
        self.batches = []

    @property
    def alphabet_size(self):
        return 2

    def membership_query(self, string):
        self.batches.append(1)
        return parity_membership(string)

    def membership_queries(self, strings):
        self.batches.append(len(strings))
        return [parity_membership(s) for s in strings]


class _CoinFlipOracle:
    """Answers by a hash of the string, so no sample size reaches significance."""

    @property
    def alphabet_size(self):
        return 2

    def membership_query(self, string):
        return hash(tuple(string)) % 2 == 0

    def membership_queries(self, strings):
        return [self.membership_query(s) for s in strings]


class _StubSampler:
    length = 20


class _StubConfig:
    # Strong enough that the derived cap falls back on the floor.
    min_signal_strength = 0.3


class _StubTable:
    # Two prefixes, one reaching each state, so both get relabelled.
    prefixes = ([0], [1])


class _StubPst:
    def __init__(self, seed=0):
        self.oracle = _CountingOracle()
        self.sampler = _StubSampler()
        self.table = _StubTable()
        self.rng = np.random.default_rng(seed)
        self.decision_boundary = 0.5
        self.config = _StubConfig()


class TestDenoiseAcceptLabels(unittest.TestCase):
    def test_corrects_a_mislabelled_state(self):
        # State 1 (odd number of 1s) is wrongly marked accepting.
        mislabelled = DFA(
            states=set(PARITY.states),
            input_symbols=set(PARITY.input_symbols),
            transitions={s: dict(PARITY.transitions[s]) for s in PARITY.states},
            initial_state=PARITY.initial_state,
            final_states={0, 1},
            allow_partial=False,
        )
        out = denoise_accept_labels(_StubPst(), mislabelled)
        self.assertEqual(set(out.final_states), {0})

    def test_leaves_correct_labels_alone(self):
        out = denoise_accept_labels(_StubPst(), PARITY)
        self.assertEqual(set(out.final_states), set(PARITY.final_states))

    def test_batching_does_not_change_the_labels(self):
        # block_size=1 reproduces the original one-at-a-time sequential test.
        one_at_a_time = denoise_accept_labels(_StubPst(), PARITY, block_size=1)
        batched = denoise_accept_labels(_StubPst(), PARITY, block_size=32)
        self.assertEqual(set(batched.final_states), set(one_at_a_time.final_states))

    def test_queries_are_issued_in_blocks(self):
        pst = _StubPst()
        denoise_accept_labels(pst, PARITY, block_size=32)
        batches = pst.oracle.batches
        self.assertTrue(batches)
        # Every call is a batch, and none of them is a lone string.
        self.assertNotIn(1, batches)


class TestDenoiseReportsAnExhaustedBudget(unittest.TestCase):
    def test_says_so_when_the_budget_decides_nothing(self):
        # The silence this replaces is how a cap below the test's requirement
        # went unnoticed: every state undecided, and nothing said so.
        pst = _StubPst()
        pst.oracle = _CoinFlipOracle()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            denoise_accept_labels(pst, PARITY)
        self.assertIn("without reaching significance", out.getvalue())

    def test_a_state_too_small_to_have_had_a_chance_is_not_named(self):
        # ALL_ONES reaches its accepting state only by the single all-1s string,
        # so it runs out of strings rather than out of budget -- the condition
        # the report turns on, and the reason it is not just "undecided".
        pst = _StubPst()
        pst.oracle = _CoinFlipOracle()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            denoise_accept_labels(pst, ALL_ONES)
        # State 1 takes every other string and runs out of budget; state 0 takes
        # only the all-1s string and never had one to run out of.
        self.assertIn("on [1] without reaching significance", out.getvalue())

    def test_quiet_when_every_state_decides(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            denoise_accept_labels(_StubPst(), PARITY)
        self.assertNotIn("without reaching significance", out.getvalue())


def _decisions(num_samples, rate, boundary, failure_prob, *, trials, seed):
    """What ``binomial_side_of_boundary`` says about a state answering at ``rate``."""
    rng = np.random.default_rng(seed)
    counts = rng.binomial(num_samples, rate, size=trials)
    return [
        binomial_side_of_boundary(
            int(c), num_samples, boundary, failure_prob=failure_prob
        )
        for c in counts
    ]


# Verifying a rate of 1 - 1e-5 needs millions of draws, so the contract is
# simulated at failure probabilities a few thousand can resolve; the size is
# searched by the same code at any of them.
CASES = [(0.3, 0.5), (0.15, 0.5), (0.1, 0.4867), (0.1, 0.55), (0.2, 0.45)]
TRIALS = 4000


class TestDenoiseSampleSizeSimulated(unittest.TestCase):
    def test_a_state_decides_at_the_size_returned(self):
        for failure_prob in (0.05, 0.01):
            for signal, boundary in CASES:
                n = denoise_sample_size(signal, boundary, failure_prob=failure_prob)
                for rate, expected in (
                    (boundary + signal, True),
                    (boundary - signal, False),
                ):
                    calls = _decisions(
                        n, rate, boundary, failure_prob, trials=TRIALS, seed=0
                    )
                    got = np.mean([c is expected for c in calls])
                    # 3 sigma of Monte Carlo error below the rate asked for.
                    floor = (
                        1
                        - failure_prob
                        - 3 * np.sqrt(failure_prob * (1 - failure_prob) / TRIALS)
                    )
                    self.assertGreater(
                        got,
                        floor,
                        f"signal {signal} boundary {boundary} fp {failure_prob} "
                        f"n {n} rate {rate}: decided {got:.4f}",
                    )

    def test_a_smaller_size_would_not_do(self):
        # Without this the size could be any large number and still pass above.
        for failure_prob in (0.05, 0.01):
            for signal, boundary in CASES:
                n = denoise_sample_size(signal, boundary, failure_prob=failure_prob)
                worst = min(
                    np.mean(
                        [
                            c is expected
                            for c in _decisions(
                                n // 2,
                                rate,
                                boundary,
                                failure_prob,
                                trials=TRIALS,
                                seed=1,
                            )
                        ]
                    )
                    for rate, expected in (
                        (boundary + signal, True),
                        (boundary - signal, False),
                    )
                )
                self.assertLess(
                    worst,
                    1 - failure_prob,
                    f"half of {n} already decides at signal {signal}",
                )

    def test_the_wrong_call_stays_inside_the_failure_probability(self):
        for failure_prob in (0.05, 0.01):
            for signal, boundary in CASES:
                n = denoise_sample_size(signal, boundary, failure_prob=failure_prob)
                for rate, wrong in (
                    (boundary + signal, False),
                    (boundary - signal, True),
                ):
                    calls = _decisions(
                        n, rate, boundary, failure_prob, trials=TRIALS, seed=2
                    )
                    self.assertLess(np.mean([c is wrong for c in calls]), failure_prob)

    def test_none_when_a_rate_leaves_the_unit_interval(self):
        # Refused before any size is tried, rather than by a search running out:
        # an out-of-range rate makes every tail nan, which would also end in None.
        with unittest.mock.patch(
            "orthogonal_dfa.l_star.statistics._decides",
            side_effect=AssertionError("searched a boundary it should have refused"),
        ):
            self.assertIsNone(denoise_sample_size(0.1, 0.95))
            self.assertIsNone(denoise_sample_size(0.1, 0.05))

    def test_the_range_the_boundary_is_clamped_to_is_sizable(self):
        # identify_cluster_around holds the boundary in [signal, 1 - signal], so
        # neither end of it may come back unsizable.
        for signal in (0.1, 0.3, 0.45):
            for boundary in (signal, 1 - signal):
                self.assertIsNotNone(denoise_sample_size(signal, boundary))

    def test_denoise_skips_what_it_cannot_size_for(self):
        for signal, boundary in ((0.1, 0.95), (0.005, 0.5)):
            pst = _StubPst()
            pst.config.min_signal_strength = signal
            pst.decision_boundary = boundary
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                result = denoise_accept_labels(pst, PARITY)
            self.assertIn("Denoise skipped", out.getvalue())
            self.assertIs(result, PARITY)


if __name__ == "__main__":
    unittest.main()
