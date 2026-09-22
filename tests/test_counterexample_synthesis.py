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
    return SimpleNamespace(indecisive=set(strings), sifter=None)


#: Enough of one for the round's boundary source, which is built but not drawn
#: on here.
_PST = SimpleNamespace(sampler=SimpleNamespace(length=4))
_DFA = SimpleNamespace(transitions={})


def _take(state, wanted, *strings):
    return _accumulate_indecisive(_PST, _resolver(*strings), _DFA, state, wanted)


def _state(held=()):
    state = _PoolState([])
    if held:
        _take(state, len(held), *held)
    return state


def _taken(state):
    """Every string the round's populations hold, in the order they arrived."""
    return [string for strings in state.held.values() for string in strings]


class TestWhatARoundTakes(unittest.TestCase):
    def test_it_takes_what_it_is_asked_for(self):
        state = _state()

        self.assertEqual(2, _take(state, 2, b"a", b"b", b"c"))
        self.assertEqual(2, len(_taken(state)))

    def test_and_no_more_than_there_is(self):
        state = _state()

        self.assertEqual(1, _take(state, 5, b"a"))
        self.assertEqual([b"a"], _taken(state))

    def test_what_the_round_already_holds_is_not_taken_again(self):
        state = _state([b"a"])

        self.assertEqual(1, _take(state, 5, b"a", b"b"))
        self.assertEqual([b"a", b"b"], sorted(_taken(state)))

    def test_asking_for_none_takes_none(self):
        state = _state()

        self.assertEqual(0, _take(state, 0, b"a"))
        self.assertEqual([], _taken(state))

    def test_two_runs_take_the_same_strings(self):
        # The cap picks a sample, so which strings it picks has to be the same
        # every run or a round is not reproducible.
        strings = [bytes([i]) for i in range(20)]
        first, second = _state(), _state()

        _take(first, 5, *strings)
        _take(second, 5, *strings)

        self.assertEqual(_taken(first), _taken(second))

    def test_what_it_takes_it_also_remembers(self):
        state = _state()

        _take(state, 2, b"a", b"b")

        self.assertEqual(state.seen, set(_taken(state)))

    def test_a_round_that_takes_twice_fills_one_population(self):
        state = _state()

        _take(state, 1, b"a")
        _take(state, 1, b"b")

        self.assertEqual(1, len(state.held), "one population for the round")


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
