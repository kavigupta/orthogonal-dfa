"""Sources of prefixes, one per population.

Each round hands the next one a source per population instead of the prefixes
themselves, so what a later round needs more of it can draw more of.
"""

import itertools
import math
import unittest
from types import SimpleNamespace

import numpy as np
from automata.fa.dfa import DFA

from orthogonal_dfa.l_star.decisions import Decisions
from orthogonal_dfa.l_star.leaf_population import LeafPopulation
from orthogonal_dfa.l_star.prefix_sources import (
    MIN_YIELD,
    StateSource,
    collect,
    state_source,
)
from orthogonal_dfa.l_star.sampler import UniformSampler


class _Counted:
    """Yields a prefix ``rate`` of the time, counting how often it is asked."""

    def __init__(self, rate, total=10**6):
        self.rate = rate
        self.calls = 0
        self._total = total

    def draw(self, _wanted):
        self.calls += 1
        keep = (self.calls * self.rate) // 1 - ((self.calls - 1) * self.rate) // 1
        return bytes([self.calls // 256, self.calls % 256]) if keep else None


class TestGivingUpOnASource(unittest.TestCase):
    def test_a_source_that_yields_is_collected(self):
        source = _Counted(1.0)
        held = collect(source, wanted=20)
        self.assertEqual(len(held), 20)
        self.assertEqual(source.calls, 20)

    def test_a_source_that_cannot_deliver_is_given_up_on(self):
        # One in fifty, well under the yield the budget waits for.
        source = _Counted(0.02)
        self.assertIsNone(collect(source, wanted=20))
        self.assertEqual(source.calls, math.ceil(20 / MIN_YIELD))

    def test_a_source_at_exactly_the_yield_survives(self):
        # The budget is 1 / MIN_YIELD draws per prefix, so a source managing
        # exactly that rate is the slowest one that still delivers.
        source = _Counted(MIN_YIELD)
        held = collect(source, wanted=100)
        self.assertIsNotNone(held)
        self.assertLessEqual(source.calls, math.ceil(100 / MIN_YIELD))

    def test_duplicates_do_not_count_toward_the_ask(self):
        class OneString:
            def draw(self, _wanted):
                return bytes([7])

        self.assertIsNone(collect(OneString(), wanted=3))


class _Tree:
    """One split at the root: the accept child is leaf 1, the reject child 0."""

    def midfix_at(self, path):
        assert path == (), path
        return b""

    def path_of(self, leaf):
        return (leaf == 1,)


class _Resolver:
    def __init__(self, population):
        self.population = population
        self.tree = _Tree()


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


#: Marks a string aimed at leaf 1 that the tree rests at leaf 0 instead.  Aiming
#: misses most of the time, so this is the ordinary case rather than a broken one.
_MISS = b"\xff"


def _misses():
    """An aim that always draws, and whose draws never settle where aimed."""
    drawn = itertools.count()
    return lambda: _MISS + bytes([next(drawn) % 256])


class TestAStateSourceServesWhatIsAlreadyThere(unittest.TestCase):
    """Aiming is how a leaf nothing has reached gets its first prefixes, not how
    it gets every prefix.  Most aims land somewhere else, and a leaf the
    population already rests strings at is read from those."""

    def _source(self, resting):
        population = LeafPopulation(
            _Tree(),
            # A miss classifies the other way, so the tree rests it at leaf 0.
            lambda strings, midfix: [not s.startswith(_MISS) for s in strings],
            harvest=lambda _string: None,
            decisions=Decisions(),
        )
        for prefix in resting:
            population.add(prefix, at=(True,))
        return StateSource(_Resolver(population), 1, _misses())

    def test_a_leaf_with_members_yields_them_when_every_aim_misses(self):
        resting = [bytes([1, i]) for i in range(20)]
        drawn = collect(self._source(resting), wanted=20)
        self.assertIsNotNone(drawn)
        self.assertEqual(sorted(drawn), sorted(resting))

    def test_a_leaf_short_of_what_is_wanted_is_given_up_on(self):
        # Every aim missing and too few resting: the population is not one to
        # hold to a rate, which is what the indecisive strings are for.
        self.assertIsNone(collect(self._source([bytes([1, 0])]), wanted=20))

    def test_it_aims_before_serving_what_rests(self):
        """Aiming is what puts new strings in front of the tree, so a leaf that
        can be aimed at is aimed at even when it has members to hand."""
        reachable = DFA(
            states={0, 1},
            input_symbols={0, 1},
            transitions={0: {0: 0, 1: 1}, 1: {0: 1, 1: 1}},
            initial_state=0,
            final_states={1},
        )
        # Length 8, so aiming has room to land somewhere the leaf does not
        # already hold; the point is that it aims at all.
        population = LeafPopulation(
            _Tree(),
            lambda strings, midfix: [True] * len(strings),
            harvest=lambda _string: None,
            decisions=Decisions(),
        )
        resting = [bytes([1, i, 0, 0, 0, 0, 0, 0]) for i in range(20)]
        for prefix in resting:
            population.add(prefix, at=(True,))
        source = state_source(_Pst(8), _Resolver(population), reachable, 1)

        drawn = collect(source, wanted=20)
        self.assertTrue(
            set(drawn) - set(resting),
            "drawing served resting members without aiming anything new",
        )

    def test_each_member_is_served_once(self):
        source = self._source([bytes([1, 0]), bytes([1, 1])])
        served = [source.draw(2), source.draw(2)]
        self.assertEqual(len([x for x in served if x]), 2)
        self.assertIsNone(source.draw(2))


if __name__ == "__main__":
    unittest.main()


#: State 2 is entered only on a ``1``, so weights that never draw one never
#: reach it however long the string.
_ONE_WAY_IN = DFA(
    states={0, 1, 2},
    input_symbols={0, 1},
    transitions={0: {0: 0, 1: 2}, 1: {0: 1, 1: 1}, 2: {0: 2, 1: 2}},
    initial_state=0,
    final_states={2},
)


class _Weighted(_Pst):
    """A sampler whose symbol weights the caller chooses."""

    def __init__(self, length, weights):
        super().__init__(length)
        self.sampler = SimpleNamespace(length=length, symbol_weights=lambda _n: weights)


class TestALeafNothingReachesGetsNoSource(unittest.TestCase):
    """Not a state the round came up short on: no draw of the sampler's length
    arrives at it, so more sampling is not the answer to it."""

    def _made(self, pst, dfa, leaf):
        population = LeafPopulation(
            _Tree(),
            lambda strings, midfix: [True] * len(strings),
            harvest=lambda _string: None,
            decisions=Decisions(),
        )
        return state_source(pst, _Resolver(population), dfa, leaf)

    def test_a_state_nothing_enters_has_no_source(self):
        self.assertIsNone(self._made(_Pst(4), _UNREACHABLE, 1))

    def test_the_sampler_weights_decide_it_too(self):
        # There is a path into state 2, but not one these weights would draw.
        self.assertIsNone(self._made(_Weighted(4, [1.0, 0.0]), _ONE_WAY_IN, 2))
        self.assertIsNotNone(self._made(_Weighted(4, [0.5, 0.5]), _ONE_WAY_IN, 2))

    def test_a_length_can_put_a_state_out_of_reach(self):
        # Nothing reaches anywhere but the initial state in zero steps.
        self.assertIsNone(self._made(_Weighted(0, [0.5, 0.5]), _ONE_WAY_IN, 2))
