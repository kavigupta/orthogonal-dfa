"""The round loop's bookkeeping: what a round takes, and when it gives up."""

import unittest
from types import SimpleNamespace

from orthogonal_dfa.l_star.counterexample_synthesis import (
    STALL_PATIENCE,
    _accumulate_indecisive,
    _PoolState,
    _StallDetector,
)


def _resolver(*strings):
    return SimpleNamespace(indecisive=set(strings))


def _state(held=()):
    state = _PoolState([])
    for string in held:
        state.seen.add(string)
        state.accumulated.append(string)
    return state


class TestWhatARoundTakes(unittest.TestCase):
    def test_it_takes_what_it_is_asked_for(self):
        state = _state()

        self.assertEqual(
            2, _accumulate_indecisive(_resolver(b"a", b"b", b"c"), state, 2)
        )
        self.assertEqual(2, len(state.accumulated))

    def test_and_no_more_than_there_is(self):
        state = _state()

        self.assertEqual(1, _accumulate_indecisive(_resolver(b"a"), state, 5))
        self.assertEqual([b"a"], state.accumulated)

    def test_what_the_round_already_holds_is_not_taken_again(self):
        state = _state([b"a"])

        self.assertEqual(1, _accumulate_indecisive(_resolver(b"a", b"b"), state, 5))
        self.assertEqual([b"a", b"b"], sorted(state.accumulated))

    def test_asking_for_none_takes_none(self):
        state = _state()

        self.assertEqual(0, _accumulate_indecisive(_resolver(b"a"), state, 0))
        self.assertEqual([], state.accumulated)

    def test_two_runs_take_the_same_strings(self):
        # The cap picks a sample, so which strings it picks has to be the same
        # every run or a round is not reproducible.
        strings = [bytes([i]) for i in range(20)]
        first, second = _state(), _state()

        _accumulate_indecisive(_resolver(*strings), first, 5)
        _accumulate_indecisive(_resolver(*strings), second, 5)

        self.assertEqual(first.accumulated, second.accumulated)

    def test_what_it_takes_it_also_remembers(self):
        state = _state()

        _accumulate_indecisive(_resolver(b"a", b"b"), state, 2)

        self.assertEqual(state.seen, set(state.accumulated))


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
