"""The round loop's bookkeeping: what a round takes, and when it gives up."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from orthogonal_dfa.l_star.counterexample_synthesis import (
    STALL_PATIENCE,
    BestRound,
    _accumulate_harvest,
    _counterexample_search,
    _publish_pool,
    _StallDetector,
)
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


class TestFnrHarvest(unittest.TestCase):
    _ACCEPTABLE_FNR = 0.05

    def _harvest_multiple(self, *, flag, fnr):
        pst = SimpleNamespace(
            config=SimpleNamespace(
                acceptable_fnr=self._ACCEPTABLE_FNR,
                counterexample_fnr_harvest=flag,
                counterexample_member_fallback=False,
            ),
            table=SimpleNamespace(representative=None, suffix=lambda i: b""),
            sampler=SimpleNamespace(length=10),
            compute_decision=lambda vs, representative: None,
            fnr_from_decision=lambda decision: (fnr, None),
        )
        with mock.patch.multiple(
            "orthogonal_dfa.l_star.counterexample_synthesis",
            CounterexampleSearch=mock.DEFAULT,
            SuffixFamily=mock.DEFAULT,
            MidfixTree=mock.DEFAULT,
            warmed_draws=mock.DEFAULT,
        ) as patched:
            _counterexample_search(pst, [0], acc_threshold=0.9)
        return patched["CounterexampleSearch"].call_args.args[0].harvest_multiple

    def test_off_leaves_the_multiple_alone(self):
        self.assertEqual(5, self._harvest_multiple(flag=False, fnr=0.2))

    def test_it_scales_by_fnr_over_acceptable(self):
        self.assertAlmostEqual(20, self._harvest_multiple(flag=True, fnr=0.2))

    def test_an_fnr_within_acceptable_does_not_shrink_it(self):
        self.assertEqual(5, self._harvest_multiple(flag=True, fnr=0.01))
