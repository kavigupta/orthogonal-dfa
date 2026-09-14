"""The floor on boundary strings, and what keeps it from being a demand.

Topping up toward a floor the population already meets is what turned a
saturated tree into an endless supply of strings that resolve nothing.
"""

import unittest

from orthogonal_dfa.l_star.counterexample_synthesis import _PoolState, _top_up_boundary


class _Exploding:
    """Anything a source would reach for."""

    def __getattr__(self, name):
        raise AssertionError(f"probed for {name} with the floor already met")


class TestTheFloorIsNotADemand(unittest.TestCase):
    def _state(self, held):
        state = _PoolState([])
        state.accumulated = list(held)
        state.seen = set(held)
        return state

    def test_a_population_at_the_floor_is_not_probed_for(self):
        state = self._state([bytes([i]) for i in range(3)])

        _top_up_boundary(_Exploding(), _Exploding(), _Exploding(), state, 3)

        self.assertEqual(len(state.accumulated), 3)


if __name__ == "__main__":
    unittest.main()
