"""The prefix populations the FNR limit is stated over, and growing one.

The learner reads its rate over several populations of prefixes at once -- the
uniform pool, the harvested boundary strings, and one per state -- holds each to
the limit on its own, and grows whichever one the rate belongs to.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from orthogonal_dfa.l_star.mask_table import UNIFORM, MaskTable
from orthogonal_dfa.l_star.prefix_populations import (
    PoolState,
    grow_population,
    prefixes_for_split,
)
from orthogonal_dfa.l_star.prefix_suffix_tracker import PrefixSuffixTracker

#: Anything: the table below is never asked a membership question.
_NO_ORACLE = None
#: The table is built with a population and then scoped to the test's own.
_SCRATCH = "scratch"
_KEY = b"key"


def _table(prefixes, populations):
    """A table holding ``prefixes``, scoped to ``label -> its members``."""
    table = MaskTable(_NO_ORACLE, prefixes, population=_SCRATCH)
    for label, members in populations.items():
        table.add_prefixes(members, population=label)
    table.drop_population(_SCRATCH)
    return table


def _tracker(table):
    """A tracker holding ``table``, enough of one to read a decision vector
    against a band of 0.4 to 0.6."""
    pst = PrefixSuffixTracker.__new__(PrefixSuffixTracker)
    pst.table = table
    pst.decision_boundary = 0.5
    pst.evidence_margin = 0.1
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
        self.rng = np.random.default_rng(0)

    def sample_more_prefixes(self):
        self.drawn_uniformly += 1


def _holding(label, source, prefixes=()):
    state = PoolState([], _KEY)
    state.held[label] = list(prefixes)
    state.seen.update(prefixes)
    state.sources[label] = source
    return state


class TestGrowingThePopulationTheRateBelongsTo(unittest.TestCase):
    def test_the_uniform_pool_grows_the_way_the_learner_always_grew_it(self):
        pst, state = _Pst(), PoolState([], _KEY)

        self.assertTrue(grow_population(pst, state, UNIFORM))
        self.assertEqual(1, pst.drawn_uniformly)

    def test_a_population_with_a_source_draws_through_it(self):
        # As many as the uniform draw it stands in for, so a named population
        # is not outgrown by the fallback for the ones nothing names.
        pst = _Pst()
        state = _holding(("state", 0), _Source(_words(100), worth=True))

        self.assertTrue(grow_population(pst, state, ("state", 0)))

        self.assertEqual(_A_DRAW, len(state.held[("state", 0)]))
        self.assertEqual(
            sorted(state.held[("state", 0)]), pst.table.added[("state", 0)]
        )

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

    def test_and_its_anchor_is_dropped(self):
        pst = _Pst()
        state = _holding(("boundary", 1), _Source([], worth=False), _words(3))
        state.anchored[("boundary", 1)] = _words(1)[0]

        grow_population(pst, state, ("boundary", 1))

        self.assertEqual({}, state.anchored)

    def test_a_population_this_round_has_no_source_for_is_retired_too(self):
        pst, state = _Pst(), PoolState([], _KEY)
        state.held[("state", 9)] = _words(3)

        self.assertFalse(grow_population(pst, state, ("state", 9)))
        self.assertEqual([("state", 9)], pst.table.dropped)


class _ProvenSource(_Source):
    proven = True


class TestCertificationReserve(unittest.TestCase):
    def test_a_string_has_one_side_in_every_population_and_draw(self):
        words = _words(200)
        state, other = PoolState([], _KEY), PoolState([], _KEY)
        sides = [state.in_reserve(w) for w in words]

        self.assertEqual(sides, [other.in_reserve(w) for w in words])
        self.assertEqual(sides, [state.in_reserve(w) for w in words])
        self.assertTrue(0 < sum(sides) < len(words))
        for label in (("state", 0), ("state", 1)):
            state.hold(label, _ProvenSource(_words(200), worth=True), _A_DRAW)
        for label in (("state", 0), ("state", 1)):
            self.assertTrue(all(state.in_reserve(w) for w in state.reserves[label]))
            self.assertFalse(any(state.in_reserve(w) for w in state.held[label]))

    def test_a_tabled_string_never_enters_a_reserve(self):
        state = PoolState([], _KEY)
        words = _words(300)
        reserve_side = [w for w in words if state.in_reserve(w)][:10]
        state.take(reserve_side[0], None)
        state.held[("state", 1)] = [reserve_side[1]]
        label = ("state", 0)

        state.hold(label, _ProvenSource(words, worth=True), 20)
        self.assertFalse(set(reserve_side[:2]) & set(state.reserves[label]))
        self.assertTrue(state.reserves[label])

        state.sources[label] = _ProvenSource(list(words), worth=True)
        got = prefixes_for_split(None, state, label, 20)
        self.assertFalse(set(reserve_side[:2]) & set(got))

    def test_the_key_changes_the_sides(self):
        words = _words(200)
        a, b = PoolState([], b"a"), PoolState([], b"b")
        self.assertNotEqual(
            [a.in_reserve(w) for w in words], [b.in_reserve(w) for w in words]
        )

    def test_the_table_and_the_reserve_split_one_run_of_draws(self):
        pst = _Pst()
        state = PoolState([], _KEY)
        label = ("state", 0)
        state.hold(label, _ProvenSource(_words(1000), worth=True), _A_DRAW)
        grow_population(pst, state, label)

        in_table = set(state.held[label])
        self.assertEqual(2 * _A_DRAW, len(in_table))
        self.assertTrue(state.reserves[label])
        self.assertFalse(in_table & set(state.reserves[label]))

    def test_the_gate_is_served_from_the_reserve_first(self):
        state = PoolState([], _KEY)
        label = ("state", 0)
        state.sources[label] = _ProvenSource(_words(100), worth=True)
        kept = [w for w in _words(100, offset=500) if state.in_reserve(w)][:3]
        state.reserves[label] = list(kept)

        got = prefixes_for_split(None, state, label, 5)

        self.assertEqual(kept, got[:3])
        self.assertEqual(5, len(got))
        self.assertEqual([], state.reserves[label])

    def test_the_gate_s_fresh_draws_are_all_reserve_side(self):
        state = PoolState([], _KEY)
        label = ("state", 0)
        state.sources[label] = _ProvenSource(_words(1000), worth=True)
        state.reserves[label] = []

        got = prefixes_for_split(None, state, label, 20)

        self.assertEqual(20, len(got))
        self.assertTrue(all(state.in_reserve(w) for w in got))


if __name__ == "__main__":
    unittest.main()
