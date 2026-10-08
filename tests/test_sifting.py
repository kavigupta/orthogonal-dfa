"""Walking a probe from its start against the tree, and narrowing to where they
disagree."""

import unittest

from orthogonal_dfa.l_star.midfix_tree import MidfixTree
from orthogonal_dfa.l_star.sifting import (
    ANCHOR,
    EDGE,
    END,
    SEARCH,
    Block,
    Sifter,
    check_from,
    first_disagreeing_edge,
    read_from,
    walk_from,
)

#: Every state steps to 1, so a walk of any non-empty probe ends there.
_STEPS_TO_ONE = {0: {0: 1, 1: 1}, 1: {0: 1, 1: 1}}
#: Nothing is learned out of state 1 on a 0.
_OPEN_AT_ONE = {0: {0: 1, 1: 1}, 1: {1: 1}}
_PROBE = bytes([0, 1, 0, 1])

_PLACES_EVERYTHING = lambda seq: 0


def _sift(places):
    """A sift by ``places``, with ``seq + b"?"`` as the boundary of what it
    cannot place."""

    def sift(seq):
        leaf = places(seq)
        return (leaf, None) if leaf is not None else (None, seq + b"?")

    return sift


class TestWalkingFromTheStart(unittest.TestCase):
    def test_it_follows_the_learned_edges_from_where_the_start_sifts(self):
        states, block = walk_from(_PROBE, _sift(lambda seq: 0), _STEPS_TO_ONE, 2)

        self.assertEqual([None, None, 0, 1, 1], states)
        self.assertIsNone(block)

    def test_a_start_the_cut_cannot_place_blocks_with_its_boundary(self):
        states, block = walk_from(_PROBE, _sift(lambda seq: None), _STEPS_TO_ONE, 2)

        self.assertIsNone(states)
        self.assertEqual(Block(ANCHOR, 2, _PROBE[:2] + b"?"), block)

    def test_an_open_edge_into_a_prefix_the_cut_cannot_place_leaves_its_boundary(
        self,
    ):
        places = lambda seq: None if len(seq) == 4 else 0
        states, block = walk_from(_PROBE, _sift(places), _OPEN_AT_ONE, 1)

        # 0 -1-> 1 -0-> open: blocked at the edge out of index 2.
        self.assertEqual([None, 0, 1], states)
        self.assertEqual(Block(EDGE, 3, None), block)
        states, block = walk_from(_PROBE + b"\x00", _sift(places), {0: {}}, 3)
        self.assertEqual(Block(EDGE, 4, _PROBE + b"?"), block)

    def test_an_open_edge_from_a_prefix_sifting_to_its_state_leaves_that_prefix(
        self,
    ):
        places = lambda seq: 1 if len(seq) == 2 else 0
        _, block = walk_from(_PROBE, _sift(places), _OPEN_AT_ONE, 1)

        self.assertEqual(Block(EDGE, 3, None, _PROBE[:2]), block)
        self.assertEqual(_PROBE[:2], block.found)

    def test_an_open_edge_from_a_prefix_the_cut_cannot_place_leaves_its_boundary(
        self,
    ):
        places = lambda seq: None if len(seq) == 2 else 0
        _, block = walk_from(_PROBE, _sift(places), _OPEN_AT_ONE, 1)

        self.assertEqual(Block(EDGE, 3, _PROBE[:2] + b"?"), block)

    def test_an_open_edge_from_a_prefix_sifting_elsewhere_leaves_nothing(self):
        _, block = walk_from(_PROBE, _sift(lambda seq: 0), _OPEN_AT_ONE, 1)

        self.assertEqual(EDGE, block.kind)
        self.assertIsNone(block.found)


class TestReadingFromTheStart(unittest.TestCase):
    def test_a_whole_probe_the_cut_cannot_place_blocks_at_its_end(self):
        places = lambda seq: None if len(seq) == 4 else 0
        _, block, disagrees = read_from(_PROBE, _sift(places), _STEPS_TO_ONE, 2)

        self.assertEqual(Block(END, 4, _PROBE + b"?"), block)
        self.assertFalse(disagrees)

    def test_it_says_where_the_walk_and_the_sift_disagree(self):
        _, block, disagrees = read_from(
            _PROBE, _sift(_PLACES_EVERYTHING), _STEPS_TO_ONE, 2
        )

        self.assertIsNone(block)
        self.assertTrue(disagrees)

    def test_a_search_places_what_the_cut_cannot_at_the_middle(self):
        # The walk reads 0, 0, 1, 1, 1 and the cut 0 throughout but for the
        # two-symbol prefix, which only the middle places; where it sends that
        # prefix decides which edge the search lands on.
        places = lambda seq: None if len(seq) == 2 else 0
        delta = {0: {0: 0, 1: 1}, 1: {0: 1, 1: 1}}

        for middle, edge in ((1, 3), (0, 2)):
            _, block, fd = check_from(
                _PROBE, _sift(places), lambda seq, m=middle: m, delta, 0
            )
            self.assertIsNone(block)
            self.assertEqual(edge, fd)

    def test_a_tie_at_the_middle_blocks_the_search(self):
        places = lambda seq: None if len(seq) == 2 else 0
        _, block, fd = check_from(
            _PROBE, _sift(places), lambda seq: None, _STEPS_TO_ONE, 0
        )

        self.assertEqual(Block(SEARCH, 2, _PROBE[:2] + b"?"), block)
        self.assertIsNone(fd)


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


class _Middle:
    """Places everything on the root's accept side and nothing below it, and
    reads the middle of the band as ``side``."""

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


if __name__ == "__main__":
    unittest.main()
