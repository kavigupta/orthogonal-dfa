"""Reading a probe from its start against the tree."""

import unittest

from orthogonal_dfa.l_star.sifting import (
    AGREE,
    EDGE,
    END_UNDECIDED,
    PAIR,
    START_UNDECIDED,
    TRIPLE,
    UNLEARNED_EDGE,
    Outcome,
    read,
)

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


def _read(places, transitions, k, probe=_PROBE):
    return read(probe, _sift(places), transitions, k)


class TestReadingAProbe(unittest.TestCase):
    def test_a_walk_ending_where_the_whole_probe_sifts_agrees(self):
        self.assertEqual(AGREE, _read(lambda seq: 1, _STEPS_TO_ONE, 2).kind)

    def test_a_start_the_cut_cannot_place_cuts_the_read_short(self):
        self.assertEqual(
            Outcome(START_UNDECIDED, 2, _PROBE[:2] + b"?", None),
            _read(lambda seq: None, _STEPS_TO_ONE, 2),
        )

    def test_a_whole_probe_the_cut_cannot_place_cuts_the_read_short(self):
        places = lambda seq: None if len(seq) == 4 else 0
        self.assertEqual(
            Outcome(END_UNDECIDED, 4, _PROBE + b"?", None),
            _read(places, _STEPS_TO_ONE, 2),
        )

    def test_an_unlearned_edge_cut_short_either_side_ends_the_walk_there(self):
        # 0 -1-> 1 -0-> unlearned: the prefixes either side are 3 and 2 long.
        for undecided in (3, 2):
            places = lambda seq, u=undecided: None if len(seq) == u else 0
            self.assertEqual(
                Outcome(END_UNDECIDED, undecided, _PROBE[:undecided] + b"?", None),
                _read(places, _OPEN_AT_ONE, 1),
            )

    def test_an_unlearned_edge_leaves_the_prefix_before_it_where_it_sifts_there(
        self,
    ):
        places = lambda seq: 1 if len(seq) == 2 else 0
        self.assertEqual(
            Outcome(UNLEARNED_EDGE, 2, _PROBE[:2], 1), _read(places, _OPEN_AT_ONE, 1)
        )

    def test_an_unlearned_edge_after_a_wrong_one_is_searched_back_to_it(self):
        # The walk reaches 1 after two symbols, where the cut places them at 0.
        self.assertEqual(
            Outcome(EDGE, 2, None, 0), _read(lambda seq: 0, _OPEN_AT_ONE, 1)
        )


#: The walk steps from state i to i + 1, so it is at state i after i symbols.
_COUNTING = {i: {0: i + 1} for i in range(8)}
_LONG = bytes(8)


def _search(undecided, flip=4):
    """Reads ``_LONG`` against a tree that agrees with the walk below ``flip``
    symbols and disagrees from it on, and places nowhere the prefixes whose
    lengths are in ``undecided``."""

    def places(seq):
        if len(seq) in undecided:
            return None
        return len(seq) if len(seq) < flip else -1

    return read(_LONG, _sift(places), _COUNTING, 0)


class TestSearchingWhereTheyPart(unittest.TestCase):
    def test_decided_reads_narrow_to_an_edge(self):
        self.assertEqual((EDGE, 4), _search(set())[:2])

    def test_an_undecided_read_between_agree_and_disagree_is_a_triple(self):
        self.assertEqual(Outcome(TRIPLE, 4, bytes(4) + b"?", None), _search({4}))

    def test_two_adjacent_undecided_reads_are_a_pair(self):
        self.assertEqual((PAIR, 3), _search({3, 4})[:2])

    def test_an_undecided_read_away_from_the_edge_is_stepped_past(self):
        # Prefix 4 is the first mid; its decided neighbours say the edge is above.
        self.assertEqual((EDGE, 6), _search({4}, flip=6)[:2])


if __name__ == "__main__":
    unittest.main()
