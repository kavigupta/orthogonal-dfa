"""The round loop's bookkeeping: what a round takes, and when it gives up."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from orthogonal_dfa.l_star import counterexample_synthesis as cs
from orthogonal_dfa.l_star.counterexample_synthesis import (
    STALL_PATIENCE,
    _publish_pool,
    _StallDetector,
)
from orthogonal_dfa.l_star.prefix_populations import PoolState


def _state():
    return PoolState([])


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


class TestWhatARoundHolds(unittest.TestCase):
    def test_it_holds_what_it_has_not_seen_under_a_label_of_its_own(self):
        state = _state()
        state.seen.add(b"a")

        state.hold_found("check", [b"a", b"b"], source="source")
        state.hold_found("check", [b"c"], source="other")

        self.assertEqual({("check", 1): [b"b"], ("check", 2): [b"c"]}, state.held)
        self.assertEqual({b"a", b"b", b"c"}, state.seen)

    def test_nothing_new_holds_nothing(self):
        state = _state()
        state.seen.add(b"a")

        state.hold_found("check", [b"a"], source="source")

        self.assertEqual({}, state.held)


class TestWhenARoundGivesUp(unittest.TestCase):
    def _rounds_of(self, *, states, improved, settled):
        """One verdict per round, for a run of identical rounds."""
        stall = _StallDetector(STALL_PATIENCE)
        return [
            stall.stalled(states=states, improved=improved, settled=settled)
            for _ in range(STALL_PATIENCE + 1)
        ]

    def test_a_round_that_shows_nothing_runs_out_of_patience(self):
        rounds = self._rounds_of(states=2, improved=False, settled=lambda: True)

        self.assertEqual(STALL_PATIENCE, rounds.index(True))

    def test_a_new_state_is_progress(self):
        stall = _StallDetector(STALL_PATIENCE)

        never = [
            stall.stalled(states=n, improved=False, settled=lambda: True)
            for n in range(2, 2 + STALL_PATIENCE + 1)
        ]

        self.assertNotIn(True, never)

    def test_a_better_hypothesis_is_progress(self):
        rounds = self._rounds_of(states=2, improved=True, settled=lambda: True)

        self.assertNotIn(True, rounds)

    def test_a_state_left_to_split_is_progress(self):
        rounds = self._rounds_of(states=2, improved=False, settled=lambda: False)

        self.assertNotIn(True, rounds)

    def test_the_count_it_keeps_is_consecutive(self):
        stall = _StallDetector(STALL_PATIENCE)
        done = lambda: True

        stall.stalled(states=2, improved=False, settled=done)
        stall.stalled(states=3, improved=False, settled=done)  # a new state resets

        self.assertFalse(stall.stalled(states=3, improved=False, settled=done))


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
                None, SimpleNamespace(num_states=1), None, state, 1
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
