"""Sources of prefixes, one per population.

Each round hands the next one a source per population instead of the prefixes
themselves, so what a later round needs more of it can draw more of.
"""

import math
import unittest

import numpy as np
from automata.fa.dfa import DFA

from orthogonal_dfa.l_star.decisions import Decisions
from orthogonal_dfa.l_star.leaf_population import LeafPopulation
from orthogonal_dfa.l_star.prefix_sources import MIN_YIELD, StateSource, collect
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


class TestAStateSourceServesWhatIsAlreadyThere(unittest.TestCase):
    """Aiming is how a leaf nothing has reached gets its first prefixes, not how
    it gets every prefix.  A leaf the population already rests strings at can be
    read without drawing anything."""

    def _source(self, resting):
        population = LeafPopulation(
            _Tree(),
            lambda strings, midfix: [True] * len(strings),
            harvest=lambda _string: None,
            decisions=Decisions(),
        )
        for prefix in resting:
            population.add(prefix, at=(True,))
        return StateSource(_Pst(2), _Resolver(population), _UNREACHABLE, 1)

    def test_a_leaf_with_members_yields_them_though_nothing_can_be_aimed(self):
        resting = [bytes([1, i]) for i in range(20)]
        drawn = collect(self._source(resting), wanted=20)
        self.assertIsNotNone(drawn)
        self.assertEqual(sorted(drawn), sorted(resting))

    def test_a_leaf_short_of_what_is_wanted_is_given_up_on(self):
        # Nothing to aim and too few resting: the population is not one to hold
        # to a rate, which is what the indecisive strings are for.
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
        source = StateSource(_Pst(8), _Resolver(population), reachable, 1)

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
