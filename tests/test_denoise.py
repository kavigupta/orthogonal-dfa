import unittest

import numpy as np
from automata.fa.dfa import DFA

from orthogonal_dfa.l_star.lstar import denoise_accept_labels

# Accepts strings with an even number of 1s.  State 0 is the accepting one.
PARITY = DFA(
    states={0, 1},
    input_symbols={0, 1},
    transitions={0: {0: 0, 1: 1}, 1: {0: 1, 1: 0}},
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
        return sum(string) % 2 == 0

    def membership_queries(self, strings):
        self.batches.append(len(strings))
        return [sum(s) % 2 == 0 for s in strings]


class _StubSampler:
    length = 8


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


if __name__ == "__main__":
    unittest.main()
