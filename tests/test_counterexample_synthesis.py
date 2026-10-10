"""The round loop's bookkeeping: what a round takes, and when it gives up."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from orthogonal_dfa.l_star import counterexample_synthesis as cs
from orthogonal_dfa.l_star.counterexample_synthesis import (
    STALL_PATIENCE,
    _accumulate_harvest,
    _publish_pool,
    _StallDetector,
)
from orthogonal_dfa.l_star.prefix_populations import PoolState

#: What draws more like the harvest, which these tests never ask.
_REPLAY = object()


def _take(strings, state, wanted):
    return _accumulate_harvest(strings, _REPLAY, state, wanted)


def _state(held=()):
    state = PoolState([])
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
    def test_the_round_draws_over_last_rounds_states(self):
        state = _state()
        state.held[("state", 7)] = [b"x"]
        drawn = SimpleNamespace(draw=lambda: b"y")

        with mock.patch.object(cs, "aim_at", lambda *_: object()), mock.patch.object(
            cs, "state_source", lambda *_, **__: drawn
        ):
            cs._per_state_members(  # pylint: disable=protected-access
                None,
                SimpleNamespace(
                    tree=SimpleNamespace(num_states=1, path_of=lambda _: ())
                ),
                None,
                state,
                1,
            )

        self.assertEqual([("state", 0)], sorted(state.held))

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
