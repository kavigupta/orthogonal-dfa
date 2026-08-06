"""The attributes ``visualize`` reads off a real learner.

``visualize`` is exercised against a stub elsewhere, so an attribute the stub
declares and the learner does not goes unnoticed -- ``learner.access`` did
exactly that.  These assert the real class, not a duck.
"""

import unittest
from types import SimpleNamespace

from orthogonal_dfa.l_star.direct_lstar import DirectLStarLearner
from orthogonal_dfa.l_star.visualize import (
    _prefill_fn,
    _resolved_edges,
    _sift_fn,
    _tree_root,
)


class _StubTable:
    prefixes = ()

    def suffix(self, v):
        return [v]

    def membership(self, strings):
        return [1] * len(strings)


class _StubOracle:
    def membership_queries(self, strings):
        return [1] * len(strings)


def _pst():
    return SimpleNamespace(
        alphabet_size=2,
        accept_thresh=0.7,
        reject_thresh=0.3,
        decision_boundary=0.5,
        evidence_margin=0.0,
        table=_StubTable(),
        oracle=_StubOracle(),
        config=SimpleNamespace(
            split_pval=0.001, min_signal_strength=0.3, suffix_family_size=2
        ),
    )


def _learner():
    return DirectLStarLearner(_pst(), [0, 1], split_fpr=None, split_miss_rate=0.02)


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
        learner.splits.record(0, [])
        learner.splits.record(1, [1])

        self.assertEqual(learner.access, {0: [], 1: [1]})

    def test_access_omits_states_nothing_is_known_to_reach(self):
        """A leaf with no member and no pool prefix has no access string, and the
        renderer skips it rather than inventing one."""
        learner = _learner()
        learner.splits.record(0, [])

        self.assertEqual(learner.access, {0: []})

    def test_access_prefers_the_shortest_member(self):
        learner = _learner()
        for member in ([1, 0, 1], [1], [0, 0]):
            learner.splits.record(0, member)

        self.assertEqual(learner.access[0], [1])


if __name__ == "__main__":
    unittest.main()
