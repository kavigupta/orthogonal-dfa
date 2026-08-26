import contextlib
import io
import unittest

import numpy as np
from automata.fa.dfa import DFA

from orthogonal_dfa.l_star.lstar import MIN_DENOISE_SAMPLES, denoise_accept_labels
from orthogonal_dfa.l_star.statistics import (
    binomial_side_of_boundary,
    denoise_sample_size,
)
from tests.dfas import PARITY, parity_membership


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
    length = 8


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

    def test_block_fills_to_the_target(self):
        # Regression: the inner bound was re-read as the block filled, so a block
        # near the cap took only half its remaining allowance and the tail
        # degraded into a series of shrinking calls.
        pst = _StubPst()
        denoise_accept_labels(pst, PARITY, max_samples=40, block_size=16)
        for size in pst.oracle.batches:
            self.assertIn(size, (16, 8))  # a full block, or the 40 - 2*16 remainder


class TestDenoiseReportsAnExhaustedBudget(unittest.TestCase):
    def test_says_so_when_the_budget_decides_nothing(self):
        # The silence this replaces is how a cap below the test's requirement
        # went unnoticed: every state undecided, and nothing said so.
        pst = _StubPst()
        pst.oracle = _CoinFlipOracle()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            denoise_accept_labels(pst, PARITY, max_samples=64)
        self.assertIn("without reaching significance", out.getvalue())

    def test_quiet_when_every_state_decides(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            denoise_accept_labels(_StubPst(), PARITY)
        self.assertNotIn("without reaching significance", out.getvalue())


class TestDenoiseSampleSize(unittest.TestCase):
    def test_the_cap_admits_the_test_it_gates(self):
        # The old fixed 200 sat below this from signal 0.15 down, so no state
        # could ever be relabelled there.
        for signal in (0.3, 0.2, 0.15, 0.1, 0.05):
            n = max(MIN_DENOISE_SAMPLES, denoise_sample_size(signal))
            decisive = [
                i
                for i in range(1, n + 1)
                if binomial_side_of_boundary(
                    int(round((0.5 + signal) * i)), i, 0.5, failure_prob=1e-5
                )
                is True
            ]
            self.assertTrue(decisive, f"no n up to {n} decides at signal {signal}")

    def test_sized_for_power_not_just_reachability(self):
        # A state sitting exactly at 0.5 + signal decides essentially always,
        # rather than the coin flip the bare minimum would give.
        rng = np.random.default_rng(0)
        for signal in (0.15, 0.1):
            n = denoise_sample_size(signal)
            draws = rng.binomial(n, 0.5 + signal, size=2000)
            decided = np.mean(
                [
                    binomial_side_of_boundary(int(a), n, 0.5, failure_prob=1e-5) is True
                    for a in draws
                ]
            )
            self.assertGreater(decided, 0.99, f"signal {signal} decided {decided}")

    def test_weaker_signal_needs_more(self):
        sizes = [denoise_sample_size(s) for s in (0.3, 0.2, 0.15, 0.1)]
        self.assertEqual(sizes, sorted(sizes))


if __name__ == "__main__":
    unittest.main()
