"""The round loop's bookkeeping: what a round takes, and when it gives up."""

import unittest
from types import SimpleNamespace

import numpy as np

from orthogonal_dfa.l_star.counterexample_synthesis import (
    STALL_PATIENCE,
    BestRound,
    _accumulate_harvest,
    _publish_pool,
    _reroot_pick,
    _reroot_start,
    _StallDetector,
)
from orthogonal_dfa.l_star.fallbacks.score_patience import Selection
from orthogonal_dfa.l_star.hypothesis_dfa import HypothesisDFA
from orthogonal_dfa.l_star.midfix_tree import MidfixTree
from orthogonal_dfa.l_star.prefix_populations import PoolState

#: What draws more like the harvest, which these tests never ask.
_REPLAY = object()


def _take(strings, state, wanted):
    return _accumulate_harvest(strings, _REPLAY, state, wanted)


def _state(held=()):
    state = PoolState([], b"key")
    for string in held:
        state.seen.add(string)
        state.harvest().append(string)
    return state


def _taken(state):
    """Every string the round's boundary population holds."""
    return [string for strings in state.held.values() for string in strings]


class _Table:
    """Just enough of the table for `_publish_pool`: which populations it holds."""

    def __init__(self):
        self.populations = {}
        self.representative = np.zeros(0, dtype=bool)

    def drop_population(self, label):
        self.populations.pop(label, None)

    def add_prefixes(self, prefixes, *, population):
        self.populations[population] = list(prefixes)


def _published(state):
    pst = SimpleNamespace(table=_Table())
    _publish_pool(pst, state)
    return pst.table.populations


class TestWhatARoundTakes(unittest.TestCase):
    def test_it_takes_what_it_is_asked_for(self):
        state = _state()

        self.assertEqual(2, _take([b"a", b"b", b"c"], state, 2))
        self.assertEqual(2, len(_taken(state)))

    def test_and_no_more_than_there_is(self):
        state = _state()

        self.assertEqual(1, _take([b"a"], state, 5))
        self.assertEqual([b"a"], _taken(state))

    def test_what_the_round_already_holds_is_not_taken_again(self):
        state = _state([b"a"])

        self.assertEqual(1, _take([b"a", b"b"], state, 5))
        self.assertEqual([b"a", b"b"], sorted(_taken(state)))

    def test_asking_for_none_takes_none(self):
        state = _state()

        self.assertEqual(0, _take([b"a"], state, 0))
        self.assertEqual([], _taken(state))

    def test_two_runs_take_the_same_strings(self):
        # The cap picks a sample, so which strings it picks has to be the same
        # every run or a round is not reproducible.
        strings = [bytes([i]) for i in range(20)]
        first, second = _state(), _state()

        _take([*strings], first, 5)
        _take([*strings], second, 5)

        self.assertEqual(_taken(first), _taken(second))

    def test_a_round_that_takes_twice_fills_one_population(self):
        state = _state()

        _take([b"a"], state, 1)
        _take([b"a", b"b"], state, 1)

        self.assertEqual(1, len(state.held), "one population for the round")

    def test_each_round_names_a_population_of_its_own(self):
        state = _state()

        _take([b"a"], state, 5)
        _published(state)
        _take([b"b"], state, 5)

        self.assertEqual([("boundary", 1), ("boundary", 2)], sorted(state.held))

    def test_what_it_takes_it_also_remembers(self):
        state = _state()

        _take([b"a", b"b"], state, 2)

        self.assertEqual(state.seen, set(_taken(state)))

    def test_a_lone_harvested_string_is_held(self):
        state = _state()

        _take([b"a"], state, 1)

        self.assertEqual({b"a"}, set(_taken(state)))
        self.assertEqual([], state.reserves.get(("boundary", 1), []))


class TestWhenARoundGivesUp(unittest.TestCase):
    def test_rounds_that_harvest_nothing_run_out_of_patience(self):
        stall = _StallDetector(STALL_PATIENCE)
        rounds = [stall.stalled(progressed=False) for _ in range(STALL_PATIENCE + 1)]

        self.assertEqual(STALL_PATIENCE - 1, rounds.index(True))

    def test_the_count_it_keeps_is_consecutive(self):
        stall = _StallDetector(STALL_PATIENCE)

        stall.stalled(progressed=False)
        stall.stalled(progressed=True)

        self.assertFalse(stall.stalled(progressed=False))


if __name__ == "__main__":
    unittest.main()


class TestWhatARoundPublishes(unittest.TestCase):
    def test_last_rounds_states_are_not_this_rounds(self):
        state = _state()
        state.held[("state", 0)] = [b"x"]

        state.retire("state")

        self.assertEqual([], sorted(state.held))

    def test_a_population_the_round_no_longer_has_leaves_the_table(self):
        state = _state()
        state.held[("state", 0)] = [b"x"]
        pst = SimpleNamespace(table=_Table())
        _publish_pool(pst, state)

        state.held.pop(("state", 0))
        _publish_pool(pst, state)

        self.assertNotIn(("state", 0), pst.table.populations)


def _funnel_hypothesis():
    """Three leaves over a two-symbol alphabet: leaf 0 (accepting) absorbs, leaf 1
    absorbs, and leaf 2 routes symbol 1 to leaf 0 and stays on symbol 0. Started at
    leaf 2 a string reaches leaf 0 iff it holds a symbol 1; started at leaf 0 every
    string absorbs, so only leaf 2 discriminates."""
    tree = MidfixTree([b""])
    tree.split(1, b"m")
    hypothesis = HypothesisDFA(lambda _s: None, tree)
    leaf0, leaf2 = (True,), (False, False)
    edges = {
        (leaf0, 0): leaf0,
        (leaf0, 1): leaf0,
        ((False, True), 0): (False, True),
        ((False, True), 1): (False, True),
        (leaf2, 0): leaf2,
        (leaf2, 1): leaf0,
    }
    hypothesis.edges = {edge: (target, b"w") for edge, target in edges.items()}
    return hypothesis


def _contains_one(strings):
    labels = [float(1 in s) for s in strings]
    return [bytes(s) for s in strings], np.asarray(labels, dtype=float)


class _ContainsOneOracle:
    def membership_queries(self, strings):
        return [int(1 in s) for s in strings]


class _BitSampler:
    def sample(self, rng, alphabet_size):
        length = int(rng.integers(1, 4))
        return bytes(int(rng.integers(alphabet_size)) for _ in range(length))


class TestReroot(unittest.TestCase):
    _PICK = (
        _contains_one([[1], [0], [1, 0], [0, 0], [0, 1], [0, 0, 0]]),
        _contains_one([[1], [0], [0, 1], [0, 0]]),
    )

    def _pst(self, *, reroot, drawing=False):
        config = SimpleNamespace(counterexample_reroot=reroot)
        if not drawing:
            return SimpleNamespace(alphabet_size=2, config=config)
        return SimpleNamespace(
            alphabet_size=2,
            config=config,
            rng=np.random.default_rng(0),
            oracle=_ContainsOneOracle(),
            sampler=_BitSampler(),
        )

    def test_enabled_it_starts_at_the_leaf_that_scores_best_held_out(self):
        hypothesis = _funnel_hypothesis()
        dfa = hypothesis.to_dfa(2, 0)
        pst = self._pst(reroot=True)
        pick = _reroot_pick(pst, Selection(search=((), ()), pick=self._PICK))
        self.assertEqual(2, _reroot_start(pst, hypothesis, dfa, pick).initial_state)

    def test_off_it_keeps_the_given_start(self):
        hypothesis = _funnel_hypothesis()
        dfa = hypothesis.to_dfa(2, 0)
        pst = self._pst(reroot=False)
        pick = _reroot_pick(pst, Selection(search=((), ()), pick=self._PICK))
        self.assertIsNone(pick)
        self.assertIs(dfa, _reroot_start(pst, hypothesis, dfa, pick))

    def test_enabled_without_a_selection_it_draws_its_own_set(self):
        hypothesis = _funnel_hypothesis()
        dfa = hypothesis.to_dfa(2, 0)
        pst = self._pst(reroot=True, drawing=True)
        pick = _reroot_pick(pst, None)
        self.assertEqual(2, _reroot_start(pst, hypothesis, dfa, pick).initial_state)


class TestBestRound(unittest.TestCase):
    _COMMON = dict(tree=None, boundary=0.5, certified=None)

    def test_a_score_outranks_consistency(self):
        best = BestRound()
        best.consider(
            consistency=0.9, score=0.1, dfa="a", round_index=0, **self._COMMON
        )
        best.consider(
            consistency=0.1, score=0.2, dfa="b", round_index=1, **self._COMMON
        )
        self.assertEqual("b", best.dfa)

    def test_without_a_score_consistency_ranks(self):
        best = BestRound()
        best.consider(
            consistency=0.9, score=None, dfa="a", round_index=0, **self._COMMON
        )
        best.consider(
            consistency=0.1, score=None, dfa="b", round_index=1, **self._COMMON
        )
        self.assertEqual("a", best.dfa)

    def test_a_certified_round_outranks_any_score(self):
        best = BestRound()
        best.consider(
            consistency=0.9, score=0.3, dfa="a", round_index=0, **self._COMMON
        )
        certified = {**self._COMMON, "certified": "b"}
        best.consider(consistency=0.9, score=0.1, dfa="b", round_index=1, **certified)
        self.assertEqual("b", best.dfa)
