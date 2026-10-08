"""Walking a probe against the tree, and narrowing to where they disagree."""

import unittest

from orthogonal_dfa.l_star.midfix_tree import MidfixTree
from orthogonal_dfa.l_star.sifting import (
    Sifter,
    anchored_walk,
    first_disagreeing_edge,
    walk,
)

#: Every state steps to 1, so a walk of any non-empty probe ends there.
_STEPS_TO_ONE = {0: {0: 1, 1: 1}, 1: {0: 1, 1: 1}}
_PROBE = bytes([0, 1, 0, 1])

_PLACES_EVERYTHING = lambda seq: 0
_PLACES_NOTHING = lambda seq: None


class TestAnchoringAWalk(unittest.TestCase):
    def test_it_anchors_where_the_tree_first_places_a_prefix(self):
        start, states = anchored_walk(
            _PROBE, lambda seq: 0 if len(seq) >= 2 else None, _STEPS_TO_ONE, 0
        )

        self.assertEqual(2, start)
        self.assertEqual([None, None, 0, 1, 1], states)

    def test_states_below_the_anchor_are_unknown(self):
        _, states = anchored_walk(
            _PROBE, lambda seq: 0 if len(seq) >= 3 else None, _STEPS_TO_ONE, 0
        )

        self.assertEqual([None, None, None, 0, 1], states)

    def test_it_anchors_no_earlier_than_asked(self):
        start, states = anchored_walk(_PROBE, _PLACES_EVERYTHING, _STEPS_TO_ONE, 3)

        self.assertEqual(3, start)
        self.assertEqual([None, None, None, 0, 1], states)

    def test_a_probe_nothing_places_past_the_earliest_anchors_nowhere(self):
        self.assertEqual(
            (None, None),
            anchored_walk(
                _PROBE, lambda seq: 0 if len(seq) < 2 else None, _STEPS_TO_ONE, 2
            ),
        )

    def test_a_probe_nothing_places_anchors_nowhere(self):
        self.assertEqual(
            (None, None), anchored_walk(_PROBE, _PLACES_NOTHING, _STEPS_TO_ONE, 0)
        )

    def test_an_empty_probe_anchors_nowhere(self):
        self.assertEqual(
            (None, None), anchored_walk(b"", _PLACES_EVERYTHING, _STEPS_TO_ONE, 0)
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


class _Middle:
    """Places nothing below the root's accept side, and reads the middle of the
    band as ``side``; the root itself places everything on its accept side."""

    def __init__(self, side):
        self.side = side

    def is_accept(self, _seq, midfix):
        return True if midfix == b"" else None

    def middle_side(self, _seq, _midfix):
        return self.side


class TestTheGatesReading(unittest.TestCase):
    def _sifter(self, side):
        tree = MidfixTree([b""])
        tree.split(0, b"x")
        return Sifter(tree, _Middle(side))

    def test_it_takes_the_middles_side_past_a_node_the_cut_cannot_place(self):
        self.assertEqual((0, [b"sx"]), self._sifter(True).halfway(b"s"))
        self.assertEqual((2, [b"sx"]), self._sifter(False).halfway(b"s"))

    def test_a_tie_reaches_no_leaf(self):
        self.assertEqual((None, [b"sx"]), self._sifter(None).halfway(b"s"))

    def test_its_reads_are_not_the_passs(self):
        sifter = self._sifter(True)
        sifter.halfway(b"s")
        self.assertEqual(0, sifter.reads)


class TestWalking(unittest.TestCase):
    def test_a_walk_follows_the_transitions_from_its_start(self):
        self.assertEqual([0, 1, 1], walk(bytes([0, 1]), 0, _STEPS_TO_ONE))
