"""The attributes ``visualize`` reads off a real learner.

``visualize`` is exercised against a stub elsewhere, so an attribute the stub
declares and the learner does not goes unnoticed -- ``learner.access`` did
exactly that.  These assert the real class, not a duck.
"""

import unittest

from orthogonal_dfa.l_star.direct_lstar import DirectLStarLearner
from orthogonal_dfa.l_star.visualize import (
    _prefill_fn,
    _resolved_edges,
    _sift_fn,
    _tree_root,
)
from tests.direct_lstar_stubs import make_pst


def _learner():
    return DirectLStarLearner(make_pst(), [0, 1])


class TestVisualizeDuckType(unittest.TestCase):
    def test_every_accessor_visualize_uses_resolves(self):
        learner = _learner()
        self.assertTrue(callable(_sift_fn(learner)))
        self.assertTrue(callable(_prefill_fn(learner)))
        self.assertIsNotNone(_tree_root(learner))
        self.assertIsNotNone(_resolved_edges(learner))
        for attr in ("access", "num_states", "pst"):
            self.assertTrue(hasattr(learner, attr), attr)

    def test_access_maps_states_to_strings(self):
        learner = _learner()
        learner.population.add([], at=learner.tree.path_of(0))
        learner.population.add([1], at=learner.tree.path_of(1))

        self.assertEqual(learner.access, {0: [], 1: [1]})

    def test_access_omits_states_nothing_is_known_to_reach(self):
        """A leaf with no member and no pool prefix has no access string, and the
        renderer skips it rather than inventing one."""
        learner = _learner()
        learner.population.add([], at=learner.tree.path_of(0))

        self.assertEqual(learner.access, {0: []})

    def test_access_prefers_the_shortest_member(self):
        # Leaf 1: the empty string reaches the initial state (leaf 0), so it does
        # not shadow the shortest recorded member here.
        learner = _learner()
        for member in ([1, 0, 1], [1], [0, 0]):
            learner.population.add(member, at=learner.tree.path_of(1))

        self.assertEqual(learner.access[1], [1])


if __name__ == "__main__":
    unittest.main()
