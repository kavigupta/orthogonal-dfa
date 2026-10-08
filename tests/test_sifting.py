"""Walking a probe from its start against the tree, and narrowing to where they
disagree."""

import unittest

from orthogonal_dfa.l_star.midfix_tree import MidfixTree
from orthogonal_dfa.l_star.sifting import Block, Sifter, first_disagreeing_edge, walk

#: Every state steps to 1, so a walk of any non-empty probe ends there.
_STEPS_TO_ONE = {0: {0: 1, 1: 1}, 1: {0: 1, 1: 1}}
#: Nothing is learned out of state 1 on a 0.
_OPEN_AT_ONE = {0: {0: 1, 1: 1}, 1: {1: 1}}
_PROBE = bytes([0, 1, 0, 1])


def _sift(places):
    """A sift by ``places``, with ``seq + b"?"`` as the boundary of what it
    cannot place."""

    def sift(seq):
        leaf = places(seq)
        return (leaf, None) if leaf is not None else (None, seq + b"?")

    return sift


def _walk(places, transitions, k, *, whole=True):
    return walk(_PROBE, _sift(places), transitions, k, whole=whole)


class TestWalkingFromTheStart(unittest.TestCase):
    def test_it_follows_the_learned_edges_and_sifts_the_whole_probe(self):
        self.assertEqual(
            ([None, None, 0, 1, 1], None, 0), _walk(lambda seq: 0, _STEPS_TO_ONE, 2)
        )

    def test_a_walk_alone_does_not_sift_the_whole_probe(self):
        places = lambda seq: None if seq == _PROBE else 0
        self.assertEqual(
            ([None, None, 0, 1, 1], None, None),
            _walk(places, _STEPS_TO_ONE, 2, whole=False),
        )

    def test_a_start_the_cut_cannot_place_is_left(self):
        self.assertEqual(
            ([None, None], Block(2, _PROBE[:2], True), None),
            _walk(lambda seq: None, _STEPS_TO_ONE, 2),
        )

    def test_an_open_edge_into_a_prefix_the_cut_cannot_place_leaves_it(self):
        # 0 -1-> 1 -0-> open: blocked at the edge out of index 2.
        states, block, _ = _walk(
            lambda seq: None if len(seq) == 3 else 0, _OPEN_AT_ONE, 1
        )

        self.assertEqual([None, 0, 1], states)
        self.assertEqual(Block(3, _PROBE[:3], True), block)

    def test_an_open_edge_leaves_the_prefix_before_it_where_it_sifts_there(self):
        for places, undecided in (
            (lambda seq: 1 if len(seq) == 2 else 0, False),
            (lambda seq: None if len(seq) == 2 else 0, True),
        ):
            _, block, _ = _walk(places, _OPEN_AT_ONE, 1)
            self.assertEqual(Block(3, _PROBE[:2], undecided), block)

    def test_an_open_edge_from_a_prefix_sifting_elsewhere_is_a_disagreement(self):
        # The walk reaches 1 after two symbols, where the cut places them at 0.
        self.assertEqual(([None, 0, 1], None, 0), _walk(lambda seq: 0, _OPEN_AT_ONE, 1))

    def test_a_whole_probe_the_cut_cannot_place_leaves_its_boundary(self):
        places = lambda seq: None if len(seq) == 4 else 0

        self.assertEqual(
            Block(4, _PROBE + b"?", True), _walk(places, _STEPS_TO_ONE, 2)[1]
        )


class TestNarrowingToTheEdge(unittest.TestCase):
    def test_it_finds_where_the_walk_and_the_tree_part(self):
        # The tree says 0 throughout; the walk says 1 from index 1 on, so the
        # edge they part over is the first.
        self.assertEqual(
            1, first_disagreeing_edge(_PROBE, [0, 1, 1, 1, 1], lambda seq: 0, 0, 4)
        )

    def test_an_edge_already_narrowed_to_is_returned_as_is(self):
        self.assertEqual(
            3, first_disagreeing_edge(_PROBE, [0, 0, 0, 0, 1], lambda seq: 0, 2, 3)
        )


class _Middle:
    """Places everything on the root's accept side and nothing below it, and
    reads the middle of the band as on ``side``."""

    def __init__(self, side):
        self.side = side

    def is_accept(self, _seq, midfix):
        return True if midfix == b"" else None

    def middle_side(self, _seq, _midfix):
        return self.side


class TestTheMiddleReading(unittest.TestCase):
    def _sifter(self, side):
        tree = MidfixTree([b""])
        tree.split(0, b"x")
        return Sifter(tree, _Middle(side))

    def test_it_takes_the_middles_side_past_a_node_the_cut_cannot_place(self):
        self.assertEqual(0, self._sifter(True).halfway(b"s"))
        self.assertEqual(2, self._sifter(False).halfway(b"s"))

    def test_its_reads_are_not_the_cuts(self):
        sifter = self._sifter(True)
        sifter.halfway(b"s")
        self.assertEqual(0, sifter.reads)


if __name__ == "__main__":
    unittest.main()
