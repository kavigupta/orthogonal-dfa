"""Walking a probe against the tree, and narrowing to where they disagree."""

import unittest

from orthogonal_dfa.l_star.sifting import anchored_walk, first_disagreeing_edge

#: Every state steps to 1, so a walk of any non-empty probe ends there.
_STEPS_TO_ONE = {0: {0: 1, 1: 1}, 1: {0: 1, 1: 1}}
_PROBE = bytes([0, 1, 0, 1])

_PLACES_EVERYTHING = lambda seq: 0
_PLACES_NOTHING = lambda seq: None


class TestAnchoringAWalk(unittest.TestCase):
    def test_it_anchors_where_the_tree_first_places_a_prefix(self):
        start, states = anchored_walk(
            _PROBE, lambda seq: 0 if len(seq) >= 2 else None, _STEPS_TO_ONE
        )

        self.assertEqual(2, start)
        self.assertEqual([None, None, 0, 1, 1], states)

    def test_states_below_the_anchor_are_unknown(self):
        _, states = anchored_walk(
            _PROBE, lambda seq: 0 if len(seq) >= 3 else None, _STEPS_TO_ONE
        )

        self.assertEqual([None, None, None, 0, 1], states)

    def test_a_probe_nothing_places_anchors_nowhere(self):
        self.assertEqual(
            (None, None), anchored_walk(_PROBE, _PLACES_NOTHING, _STEPS_TO_ONE)
        )

    def test_an_empty_probe_anchors_nowhere(self):
        self.assertEqual(
            (None, None), anchored_walk(b"", _PLACES_EVERYTHING, _STEPS_TO_ONE)
        )


class TestNarrowingToTheEdge(unittest.TestCase):
    def test_it_finds_where_the_walk_and_the_tree_part(self):
        # The tree says 0 throughout; the walk says 1 from index 1 on, so the
        # edge they part over is the first.
        self.assertEqual(
            1,
            first_disagreeing_edge(_PROBE, [0, 1, 1, 1, 1], _PLACES_EVERYTHING, 0, 4),
        )

    def test_an_edge_already_narrowed_to_is_returned_as_is(self):
        self.assertEqual(
            3,
            first_disagreeing_edge(_PROBE, [0, 0, 0, 0, 1], _PLACES_EVERYTHING, 2, 3),
        )

    def test_a_prefix_the_tree_cannot_place_gives_no_edge(self):
        places = lambda seq: None if len(seq) == 2 else 0

        self.assertIsNone(first_disagreeing_edge(_PROBE, [0, 0, 1, 1, 1], places, 0, 4))


if __name__ == "__main__":
    unittest.main()
