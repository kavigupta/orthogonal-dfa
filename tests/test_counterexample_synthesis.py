"""When the round loop gives up.

What a round takes is `Pools`' business on this branch, not a free function's.
"""

import unittest

from orthogonal_dfa.l_star.counterexample_synthesis import (
    STALL_PATIENCE,
    _StallDetector,
)


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
