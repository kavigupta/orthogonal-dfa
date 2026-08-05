"""Where :meth:`DirectLStarLearner.process` anchors its walk.

Driven by stubs rather than synthesis: the anchor is a property of the walk, and
the end-to-end targets that depend on it are noisy enough that a regression shows
up as a state count that also moves for unrelated reasons.
"""

import unittest
from types import SimpleNamespace

from orthogonal_dfa.l_star.direct_lstar import _RESOLVED, DirectLStarLearner


class _StubSifter:
    """Places a string only once it is at least ``places_at`` symbols long."""

    def __init__(self, places_at, state=7):
        self.places_at = places_at
        self.state = state
        self.asked = []

    def sift_and_boundary(self, seq):
        self.asked.append(tuple(seq))
        if len(seq) < self.places_at:
            return None, tuple(seq) + ("bail",)
        return self.state, None


class _StubSplits:
    def __init__(self):
        self.recorded = []

    def record(self, state, prefix):
        self.recorded.append((state, tuple(prefix)))


class _Learner(DirectLStarLearner):
    """Captures what the walk hands to the disagreement test."""

    # pylint: disable=super-init-not-called
    def __init__(self, sifter):
        self.sifter = sifter
        self.splits = _StubSplits()
        self.dfa = SimpleNamespace(access={})
        self.indecisive = set()
        self.acted = None

    def _act_on_disagreement(self, w, states, agree_point):
        self.acted = (list(w), list(states), agree_point)
        return _RESOLVED


# Two states, so a walk that follows delta visibly alternates.
DELTA = {7: {0: 8, 1: 7}, 8: {0: 7, 1: 8}}


class TestProcessAnchor(unittest.TestCase):
    def test_anchors_at_the_shortest_placeable_prefix(self):
        """The empty string is the one the family places worst, so a probe that
        cannot start there must still be walked, from deeper in."""
        learner = _Learner(_StubSifter(places_at=2))
        learner.process([0, 1, 0, 1], DELTA)

        w, states, agree_point = learner.acted
        self.assertEqual(agree_point, 2)
        self.assertEqual(w, [0, 1, 0, 1])
        # Unplaced positions stay None; the walk follows delta from the anchor.
        self.assertEqual(states, [None, None, 7, 8, 8])

    def test_records_the_anchor_not_the_empty_string(self):
        learner = _Learner(_StubSifter(places_at=2))
        learner.process([0, 1, 0, 1], DELTA)

        self.assertEqual(learner.splits.recorded, [(7, (0, 1))])
        self.assertEqual(learner.dfa.access, {7: [0, 1]})

    def test_harvests_every_prefix_it_could_not_place(self):
        learner = _Learner(_StubSifter(places_at=2))
        learner.process([0, 1, 0, 1], DELTA)

        self.assertEqual(learner.indecisive, {("bail",), (0, "bail")})

    def test_walks_from_the_empty_string_when_it_places(self):
        learner = _Learner(_StubSifter(places_at=0))
        learner.process([0, 1], DELTA)

        _, states, agree_point = learner.acted
        self.assertEqual(agree_point, 0)
        self.assertEqual(states, [7, 8, 8])

    def test_gives_up_only_when_no_prefix_places(self):
        sifter = _StubSifter(places_at=99)
        learner = _Learner(sifter)

        self.assertEqual(learner.process([0, 1, 0], DELTA), _RESOLVED)
        self.assertIsNone(learner.acted)
        # It tried every prefix before giving up, rather than only the empty one.
        self.assertEqual(sifter.asked, [(), (0,), (0, 1)])
