"""An edge into a badly read state is held apart, rolled over, or dropped."""

import itertools
import unittest
from types import SimpleNamespace

import numpy as np

from orthogonal_dfa.l_star.counterexample_synthesis import _split_edges
from orthogonal_dfa.l_star.edge_chains import (
    DROP,
    PROMOTE,
    ROLL_OVER,
    EdgeChain,
    edge_verdict,
)
from orthogonal_dfa.l_star.prefix_populations import PoolState


class _Sifter:
    """Places a string unless its last letter is in ``undecided``; one read each."""

    def __init__(self, undecided):
        self.undecided = undecided
        self.reads = 0

    def sift_and_boundary(self, seq):
        self.reads += 1
        if seq and seq[-1] in self.undecided:
            return None, seq + b"?"
        return 0, None


class _EveryFiftieth(_Sifter):
    """Undecided on every fiftieth string ending in 1: twice a 0.01 clean rate."""

    def __init__(self):
        super().__init__(set())
        self._ones = 0

    def sift_and_boundary(self, seq):
        self.reads += 1
        if seq and seq[-1] == 1:
            self._ones += 1
            if self._ones % 50 == 0:
                return None, seq + b"?"
        return 0, None


class _Counting:
    """A source drawing fresh strings 0, 1, 2, ..., each ending in ``last``."""

    def __init__(self, last=0):
        self._next = itertools.count()
        self._last = last

    def draw(self):
        return next(self._next).to_bytes(4, "big") + bytes([self._last])


class TestAChainKeepsWhatItsRoundCouldNotPlace(unittest.TestCase):
    def test_it_keeps_a_draw_only_when_its_extension_is_undecided(self):
        parent = SimpleNamespace(draw=iter([b"x", b"y"]).__next__)
        sifter = SimpleNamespace(
            sift_and_boundary=lambda seq: (None, b"") if seq == b"y\x01" else (0, None)
        )
        chain = EdgeChain(parent, b"\x01", sifter, good=0.03, poor=0.01)

        self.assertFalse(chain.attempt_draw())
        self.assertTrue(chain.attempt_draw())
        self.assertEqual([b"y"], chain.found())


class TestAnEdgeIsJudgedAgainstTheCleanRate(unittest.TestCase):
    def test_far_above_the_factor_promotes(self):
        self.assertEqual(
            PROMOTE, edge_verdict(30, 100, 0.01, failure_prob=1e-4, final=False)
        )

    def test_none_undecided_over_many_reads_drops(self):
        self.assertEqual(
            DROP, edge_verdict(0, 2000, 0.01, failure_prob=1e-4, final=False)
        )

    def test_between_the_clean_rate_and_the_factor_rolls_over(self):
        self.assertEqual(
            ROLL_OVER, edge_verdict(200, 10000, 0.01, failure_prob=1e-4, final=False)
        )

    def test_too_few_reads_to_say_waits(self):
        self.assertIsNone(edge_verdict(1, 20, 0.01, failure_prob=1e-4, final=False))


class TestARoundSortsItsEdges(unittest.TestCase):
    def _round(self, state, sifter):
        pst = SimpleNamespace(
            acceptable_fnr=0.01, alphabet_size=2, rng=np.random.default_rng(0)
        )
        resolver = SimpleNamespace(sifter=sifter)
        dfa = SimpleNamespace(transitions={})
        _split_edges(pst, resolver, dfa, state, acc_threshold=0.98)

    def _state(self):
        state = PoolState([])
        state.held[("state", 0)] = []
        state.sources[("state", 0)] = _Counting()
        return state

    def test_an_edge_into_a_badly_read_state_becomes_a_population(self):
        state = self._state()

        self._round(state, _Sifter({1}))

        self.assertIn(("edge", 1), state.held)
        self.assertTrue(state.held[("edge", 1)])
        self.assertEqual([], state.chains)

    def test_a_clean_edge_is_dropped_and_a_dropped_chain_forgotten(self):
        state = self._state()
        state.chains = [
            EdgeChain(_Counting(), b"\x01", _Sifter(set()), good=0.03, poor=0.01)
        ]

        self._round(state, _Sifter(set()))

        self.assertNotIn(("edge", 1), state.held)
        self.assertEqual([], state.chains)

    def test_an_edge_between_the_two_rates_rolls_over_onto_its_source(self):
        state = self._state()

        self._round(state, _EveryFiftieth())

        self.assertNotIn(("edge", 1), state.held)
        self.assertEqual(1, len(state.chains))
        self.assertIs(state.sources[("state", 0)], state.chains[0].parent)
        self.assertEqual(b"\x01", state.chains[0].letter)


if __name__ == "__main__":
    unittest.main()
