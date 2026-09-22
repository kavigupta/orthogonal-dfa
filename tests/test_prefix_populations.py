"""The prefix populations the FNR limit is stated over, and growing one.

The learner reads its rate over several populations of prefixes at once -- the
uniform pool, the harvested boundary strings, and one per state -- holds each to
the limit on its own, and grows whichever one the rate belongs to.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from orthogonal_dfa.l_star.counterexample_synthesis import _PoolState, grow_population
from orthogonal_dfa.l_star.mask_table import UNIFORM, MaskTable
from orthogonal_dfa.l_star.prefix_suffix_tracker import PrefixSuffixTracker

#: Anything: the table below is never asked a membership question.
_NO_ORACLE = None
#: The table is built with a population and then scoped to the test's own.
_SCRATCH = "scratch"


def _table(prefixes, populations):
    """A table holding ``prefixes``, scoped to ``label -> its members``."""
    table = MaskTable(_NO_ORACLE, prefixes, population=_SCRATCH)
    for label, members in populations.items():
        table.add_prefixes(members, population=label)
    table.drop_population(_SCRATCH)
    return table


def _tracker(table, *, boundary=0.5, margin=0.1):
    """A tracker holding ``table``, enough of one to read a decision vector."""
    pst = PrefixSuffixTracker.__new__(PrefixSuffixTracker)
    pst.table = table
    pst.config = SimpleNamespace(fnr_limit=0.02, population_fnr_limit=0.10)
    pst.decision_boundary = boundary
    pst.evidence_margin = margin
    return pst


def _words(n, offset=0):
    return [bytes([(i + offset) // 256, (i + offset) % 256]) for i in range(n)]


class TestTheRateIsPerPopulation(unittest.TestCase):
    def test_a_small_population_is_not_averaged_away(self):
        # 100 decisive prefixes and 10 indecisive ones: 9% across the union, but
        # the ten are a population of their own and every one of them straddles.
        decisive, straddling = _words(100), _words(10, offset=100)
        table = _table(
            decisive + straddling,
            {UNIFORM: decisive, ("state", 0): straddling},
        )
        pst = _tracker(table)
        # Both classes present, or the family reads as uninformative whatever
        # the populations say.
        decision = np.array([0.9] * 50 + [0.1] * 50 + [0.5] * 10)

        self.assertAlmostEqual(
            float(np.mean((decision >= 0.4) & (decision < 0.6))), 10 / 110
        )
        rate, worst = pst.fnr_from_decision(decision)
        self.assertEqual((rate, worst), (1.0, ("state", 0)))

    def test_each_population_answers_to_its_own_limit(self):
        # The pool is a sample of what the learner will meet, so its rate is an
        # estimate and is held to the tighter limit.  A state's prefixes are a
        # to-do list, and the same rate over them is not yet a fault.
        a, b = _words(20), _words(20, offset=20)
        pst = _tracker(_table(a + b, {UNIFORM: a, ("state", 3): b}))
        # A twentieth of each straddles: 0.05, over the pool's 0.02 and under
        # the 0.10 the others answer to.
        decision = np.array(
            [0.9] * 10 + [0.1] * 9 + [0.5] + [0.9] * 10 + [0.1] * 9 + [0.5]
        )

        rate, worst = pst.fnr_from_decision(decision)

        self.assertEqual(worst, UNIFORM, "the pool is the one past its limit")
        self.assertAlmostEqual(rate, 0.05)
        self.assertGreater(rate, pst.limit_for(UNIFORM))
        self.assertLess(rate, pst.limit_for(("state", 3)))

    def test_the_population_furthest_past_its_limit_is_the_worst(self):
        # The state reads higher, and is still further inside its own limit
        # than the pool is inside the pool's.
        a, b = _words(20), _words(20, offset=20)
        pst = _tracker(_table(a + b, {UNIFORM: a, ("state", 3): b}))
        # A twentieth of the pool straddles against a tenth of the state:
        # 0.05 against 0.02 is further out than 0.10 against 0.10.
        decision = np.array(
            [0.9] * 10 + [0.1] * 9 + [0.5] + [0.9] * 9 + [0.1] * 9 + [0.5] * 2
        )

        _, worst = pst.fnr_from_decision(decision)

        self.assertEqual(worst, UNIFORM)

    def test_the_rate_names_the_population_it_belongs_to(self):
        a, b = _words(20), _words(20, offset=20)
        table = _table(a + b, {UNIFORM: a, ("state", 3): b})
        pst = _tracker(table)
        # A fifth of ``b`` straddles and none of ``a`` does.
        decision = np.array([0.9] * 10 + [0.1] * 10 + [0.5] * 4 + [0.9] * 8 + [0.1] * 8)

        rate, worst = pst.fnr_from_decision(decision)
        self.assertEqual(worst, ("state", 3))
        self.assertAlmostEqual(rate, 0.2)


class _Source:
    """A source that gives out ``members`` in order, if it is worth drawing on."""

    def __init__(self, members, *, worth):
        self.members = list(members)
        self._worth = worth

    def worth_drawing(self):
        return self._worth

    def draw(self):
        return self.members.pop(0)


class _Table:
    """Enough of a table for `grow_population`: what it dropped and added."""

    def __init__(self):
        self.dropped = []
        self.added = {}

    def drop_population(self, label):
        self.dropped.append(label)

    def add_prefixes(self, prefixes, *, population):
        self.added.setdefault(population, []).extend(prefixes)


#: What a uniform draw adds, and so what growing a named population adds.
_A_DRAW = 8


class _Pst:
    """Enough of a tracker for `grow_population`: what it grew, and how."""

    def __init__(self):
        self.drawn_uniformly = 0
        self.table = _Table()
        self.config = SimpleNamespace(num_addtl_prefixes=_A_DRAW)

    def sample_more_prefixes(self):
        self.drawn_uniformly += 1


def _holding(label, source, prefixes=()):
    state = _PoolState([])
    state.held[label] = list(prefixes)
    state.seen.update(prefixes)
    state.sources[label] = source
    return state


class TestGrowingThePopulationTheRateBelongsTo(unittest.TestCase):
    def test_the_uniform_pool_grows_the_way_the_learner_always_grew_it(self):
        pst, state = _Pst(), _PoolState([])

        self.assertTrue(grow_population(pst, state, UNIFORM))
        self.assertEqual(1, pst.drawn_uniformly)

    def test_a_population_with_a_source_draws_through_it(self):
        # As many as the uniform draw it stands in for, so a named population
        # is not outgrown by the fallback for the ones nothing names.
        pst = _Pst()
        state = _holding(("state", 0), _Source(_words(100), worth=True))

        self.assertTrue(grow_population(pst, state, ("state", 0)))

        self.assertEqual(_words(_A_DRAW), state.held[("state", 0)])
        self.assertEqual(_words(_A_DRAW), pst.table.added[("state", 0)])

    def test_a_population_nothing_draws_for_is_retired(self):
        pst = _Pst()
        state = _holding(("boundary", 1), _Source([], worth=False), _words(3))

        self.assertFalse(grow_population(pst, state, ("boundary", 1)))

        self.assertEqual([("boundary", 1)], pst.table.dropped)
        self.assertEqual({}, state.held)

    def test_and_its_strings_are_forgotten_rather_than_held_aside(self):
        # A later round that strands one of these again can pool it behind a
        # source that does draw.
        pst = _Pst()
        state = _holding(("boundary", 1), _Source([], worth=False), _words(3))

        grow_population(pst, state, ("boundary", 1))

        self.assertEqual(set(), state.seen)

    def test_a_population_this_round_has_no_source_for_is_retired_too(self):
        pst, state = _Pst(), _PoolState([])
        state.held[("state", 9)] = _words(3)

        self.assertFalse(grow_population(pst, state, ("state", 9)))
        self.assertEqual([("state", 9)], pst.table.dropped)


if __name__ == "__main__":
    unittest.main()
