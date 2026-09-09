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
    gather,
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


class _Spooled:
    """A finite supply, with the give-back a real source has."""

    def __init__(self, words):
        self._words = list(words)
        self._spare = []
        self.drawn = 0

    def draw(self, _wanted):
        if self._spare:
            return self._spare.pop(0)
        if not self._words:
            return None
        self.drawn += 1
        return self._words.pop(0)

    def unused(self, drawn):
        self._spare.extend(drawn)


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

    def test_a_failed_ask_hands_its_draws_back(self):
        # Landing a draw is the expensive part, so what a collection could not
        # use goes back rather than being paid for twice.
        source = _Spooled([bytes([i]) for i in range(4)])
        self.assertIsNone(collect(source, wanted=6))
        self.assertEqual(source.drawn, 4, "and it stopped once it ran dry")
        self.assertEqual(gather(source, wanted=4), [bytes([i]) for i in range(4)])
        self.assertEqual(source.drawn, 4, "the second ask cost nothing")

    def test_gather_keeps_what_it_got(self):
        source = _Spooled([bytes([i]) for i in range(4)])
        self.assertEqual(gather(source, wanted=6), [bytes([i]) for i in range(4)])

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


class _Weighted(_Pst):
    """A sampler whose symbol weights the caller chooses."""

    def __init__(self, length, weights):
        super().__init__(length)
        self.sampler = SimpleNamespace(length=length, symbol_weights=lambda _n: weights)


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
        return StateSource(_Resolver(population), 1, _misses(), sink=lambda _s: None)

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
        source = state_source(
            _Pst(8), _Resolver(population), reachable, 1, sink=lambda _s: None
        )

        drawn = collect(source, wanted=20)
        self.assertTrue(
            set(drawn) - set(resting),
            "drawing served resting members without aiming anything new",
        )

    def test_a_leaf_nothing_reaches_gets_no_source_at_all(self):
        # Not a state the round came up short on -- a state no draw of the
        # sampler's length arrives at, which more sampling does not change.
        population = LeafPopulation(
            _Tree(),
            lambda strings, midfix: [True] * len(strings),
            harvest=lambda _string: None,
            decisions=Decisions(),
        )
        made = state_source(
            _Pst(2), _Resolver(population), _UNREACHABLE, 1, sink=lambda _s: None
        )

        self.assertIsNone(made)

    def _made(self, pst, dfa, leaf, *, lands=True):
        population = LeafPopulation(
            _Tree(),
            # ``lands`` decides whether the tree rests an aimed string where it
            # was aimed, which is the only thing that makes the leaf a source.
            lambda strings, midfix: [lands] * len(strings),
            harvest=lambda _string: None,
            decisions=Decisions(),
        )
        return state_source(pst, _Resolver(population), dfa, leaf, sink=lambda _s: None)

    def test_the_sampler_weights_decide_reachability_too(self):
        # There is a path into state 1, but not one these weights would draw.
        self.assertIsNone(self._made(_Weighted(4, [1.0, 0.0]), _ONE_WAY_IN, 1))
        self.assertIsNotNone(self._made(_Weighted(4, [0.5, 0.5]), _ONE_WAY_IN, 1))

    def test_a_length_can_put_a_state_out_of_reach(self):
        # Nothing reaches anywhere but the initial state in zero steps.
        self.assertIsNone(self._made(_Weighted(0, [0.5, 0.5]), _ONE_WAY_IN, 1))

    def test_a_leaf_the_tree_never_rests_an_aim_at_has_no_source_either(self):
        # The hypothesis reaches it and every aim lands elsewhere, so what the
        # leaf holds is what it will ever hold.
        even = _Weighted(4, [0.5, 0.5])
        self.assertIsNotNone(self._made(even, _ONE_WAY_IN, 1, lands=True))
        self.assertIsNone(self._made(even, _ONE_WAY_IN, 1, lands=False))

    def test_taking_draws_back_makes_them_servable_again(self):
        source = self._source([bytes([1, 0]), bytes([1, 1])])
        first = [source.draw(2), source.draw(2)]
        self.assertIsNone(source.draw(2), "and then it has no more")

        source.unused(first)

        self.assertEqual(sorted([source.draw(2), source.draw(2)]), sorted(first))

    def test_each_member_is_served_once(self):
        source = self._source([bytes([1, 0]), bytes([1, 1])])
        served = [source.draw(2), source.draw(2)]
        self.assertEqual(len([x for x in served if x]), 2)
        self.assertIsNone(source.draw(2))


if __name__ == "__main__":
    unittest.main()
