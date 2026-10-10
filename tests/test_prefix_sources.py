"""Sources of prefixes, one per population.

Each round hands the next one a source per population instead of the prefixes
themselves, so what a later round needs more of it can draw more of.
"""

import itertools
import unittest
from types import SimpleNamespace

import numpy as np
from automata.fa.dfa import DFA

from orthogonal_dfa.l_star.prefix_sources import HarvestSource, aim_at, state_source
from orthogonal_dfa.l_star.rejection_source import SourceDry
from orthogonal_dfa.l_star.sampler import UniformSampler


class _Pst:
    alphabet_size = 2

    def __init__(self, length):
        self.sampler = UniformSampler(length)
        self.rng = np.random.default_rng(0)


#: State 1 loops on itself and nothing enters it, so no string of any length
#: reaches it and there is nothing for the hypothesis to aim.
_UNREACHABLE = DFA(
    states={0, 1},
    input_symbols={0, 1},
    transitions={0: {0: 0, 1: 0}, 1: {0: 1, 1: 1}},
    initial_state=0,
    final_states={1},
)


#: State 1 is entered only on a ``1``, so weights that never draw one never
#: reach it however long the string.  It is the tree's leaf 1, so aims that get
#: there do settle there.
_ONE_WAY_IN = DFA(
    states={0, 1},
    input_symbols={0, 1},
    transitions={0: {0: 0, 1: 1}, 1: {0: 1, 1: 1}},
    initial_state=0,
    final_states={1},
)


class TestALeafThatRunsDryStops(unittest.TestCase):
    """A yield the aims keep clearing says nothing about what is left to draw:
    a string already served rests where it was aimed like any other."""

    def test_a_leaf_with_nothing_left_to_draw_raises(self):
        support = [bytes([9, i]) for i in range(3)]
        aims = itertools.cycle(support)
        source = state_source(lambda _: True, lambda: next(aims))

        drawn = [source.draw() for _ in range(len(support))]

        self.assertEqual(sorted(drawn), support)
        with self.assertRaisesRegex(RuntimeError, "found no new samples"):
            source.draw()


#: A chain nothing but the all-ones string walks: one string of length 8 out of
#: 256 reaches state 8, far under the 16 that the square root of them asks for.
_ONE_STRING_IN = DFA(
    states=set(range(10)),
    input_symbols={0, 1},
    transitions={**{q: {1: q + 1, 0: 9} for q in range(9)}, 9: {0: 9, 1: 9}},
    initial_state=0,
    final_states={8},
)


class _Weighted(_Pst):
    """A sampler whose symbol weights the caller chooses."""

    def __init__(self, length, weights):
        super().__init__(length)
        self.sampler = SimpleNamespace(length=length, symbol_weights=lambda _n: weights)


class TestALeafWithNothingToDrawGetsNoSource(unittest.TestCase):
    """Not a state the round came up short on: too few draws of the sampler's
    length arrive at it, so more sampling is not the answer to it."""

    def _made(self, pst, dfa, leaf, *, lands=True):
        aim = aim_at(pst, dfa, leaf)
        if aim is None:
            return None
        # ``lands`` decides whether the tree rests an aimed string where it was
        # aimed, which is the only thing that makes the leaf a source.
        return state_source(lambda _: lands, aim)

    def test_a_state_nothing_enters_has_no_source(self):
        self.assertIsNone(self._made(_Pst(4), _UNREACHABLE, 1))

    def test_the_sampler_weights_decide_it_too(self):
        # There is a path into state 1, but not one these weights would draw.
        self.assertIsNone(self._made(_Weighted(8, [1.0, 0.0]), _ONE_WAY_IN, 1))
        self.assertIsNotNone(self._made(_Weighted(8, [0.5, 0.5]), _ONE_WAY_IN, 1))

    def test_a_length_can_put_a_state_out_of_reach(self):
        # Nothing reaches anywhere but the initial state in zero steps.
        self.assertIsNone(self._made(_Weighted(0, [0.5, 0.5]), _ONE_WAY_IN, 1))

    def test_a_leaf_thin_against_its_space_is_no_pool(self):
        # Reached at all, and by so little of the space that drawing from it
        # would be drawing the same strings again.
        self.assertIsNone(self._made(_Weighted(8, [0.5, 0.5]), _ONE_STRING_IN, 8))

    def test_a_leaf_the_tree_never_rests_an_aim_at_has_no_source_either(self):
        # The hypothesis reaches it and every aim lands elsewhere, so what the
        # leaf holds is what it will ever hold.
        even = _Weighted(8, [0.5, 0.5])
        self.assertIsNotNone(self._made(even, _ONE_WAY_IN, 1, lands=True))
        self.assertIsNone(self._made(even, _ONE_WAY_IN, 1, lands=False))


class _Finds:
    """A provenance whose every fresh draw finds ``strings``."""

    def __init__(self, *strings):
        self._strings = list(strings)

    def sample(self):
        return list(self._strings)


_FOUND = b"\x00\x01?"


class TestAHarvestSourceDrawsByProvenance(unittest.TestCase):
    def _source(self, *, known):
        return HarvestSource(
            {_Finds(_FOUND): 1},
            np.random.default_rng(0),
            known=known,
            acc_threshold=0.98,
        )

    def test_a_find_is_served(self):
        source = self._source(known=())

        self.assertTrue(source.attempt_draw())
        self.assertEqual(_FOUND, source.draw())

    def test_keeping_the_same_string_again_is_not_a_find(self):
        source = self._source(known=())

        self.assertTrue(source.attempt_draw())
        self.assertFalse(source.attempt_draw(), "the second probe found nothing new")

    def test_what_the_caller_already_holds_is_not_a_find(self):
        source = self._source(known=[_FOUND])

        self.assertFalse(source.attempt_draw())

    def test_a_source_that_finds_nothing_new_stops_rather_than_probing_forever(self):
        source = self._source(known=())

        self.assertEqual(_FOUND, source.draw())

        with self.assertRaisesRegex(SourceDry, "found no new samples"):
            source.draw()

    def test_provenances_are_drawn_in_proportion_to_what_they_found(self):
        drawn = itertools.count()

        class Finds:
            def __init__(self, tag):
                self._tag = tag

            def sample(self):
                return [next(drawn).to_bytes(4, "big") + self._tag]

        often, rarely = Finds(b"a"), Finds(b"b")
        source = HarvestSource(
            {often: 3, rarely: 1},
            np.random.default_rng(0),
            known=(),
            acc_threshold=0.98,
        )
        ends = [source.draw()[-1:] for _ in range(400)]

        self.assertAlmostEqual(ends.count(b"a") / len(ends), 3 / 4, delta=0.06)


if __name__ == "__main__":
    unittest.main()
