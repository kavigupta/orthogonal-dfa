"""One probe against hand-built hypotheses and scripted reads."""

import unittest

from orthogonal_dfa.l_star.counterexample_probe import (
    Agree,
    EdgeFound,
    End,
    Member,
    Pair,
    Start,
    Triple,
    charges,
    locate,
)
from orthogonal_dfa.l_star.midfix_tree import MidfixTree
from tests.counterexample_common import EVEN, ODD, PARITY_EDGES, WRONG_ODD_0, parity_cut


def _locate(cut, edges, x):
    return locate(cut, MidfixTree([]), edges, 0, bytes(x))


def _charges(cut, edges, x):
    return charges(_locate(cut, edges, x), 0, bytes(x))


class TestOutcomes(unittest.TestCase):
    def test_a_walk_ending_where_the_probe_sifts_agrees(self):
        self.assertEqual(
            Agree(), _locate(parity_cut(()), PARITY_EDGES, [1, 0, 1]).outcome
        )

    def test_an_undecided_start(self):
        self.assertEqual(
            Start(b""), _locate(parity_cut([b""]), PARITY_EDGES, [1, 0]).outcome
        )
        self.assertEqual([b""], _charges(parity_cut([b""]), PARITY_EDGES, [1, 0]).start)

    def test_an_undecided_end(self):
        outcome = _locate(parity_cut([b"\x01\x00"]), PARITY_EDGES, [1, 0]).outcome
        self.assertEqual(End(b"\x01\x00"), outcome)

    def test_an_unlearned_edge_where_the_cut_places_its_prefix_is_a_member(self):
        outcome = _locate(parity_cut(()), {}, [1, 0]).outcome
        self.assertEqual(Member((EVEN, 1), ODD, b""), outcome)

    def test_an_unlearned_edge_whose_successor_is_undecided_is_an_end(self):
        outcome = _locate(parity_cut([b"\x01"]), {}, [1, 0]).outcome
        self.assertEqual(End(b"\x01"), outcome)

    def test_an_unlearned_edge_the_walk_misplaces_is_searched(self):
        located = _locate(parity_cut(()), {(EVEN, 1): (EVEN, b"")}, [1, 0])
        self.assertEqual(EdgeFound(EVEN, 1, ODD, b""), located.outcome)
        self.assertEqual([2, 1], list(located.sifts))

    def test_a_disagreement_is_bisected_to_its_edge(self):
        located = _locate(parity_cut(()), WRONG_ODD_0, [1, 0, 0, 0])
        self.assertEqual(EdgeFound(ODD, 0, ODD, b"\x01"), located.outcome)
        self.assertEqual([4, 2, 1], list(located.sifts))

    def test_an_undecided_middle_between_agreement_and_disagreement_is_a_triple(self):
        located = _locate(parity_cut([b"\x00\x01"]), WRONG_ODD_0, [0, 1, 0, 0])
        self.assertEqual(Triple(2, b"\x00\x01"), located.outcome)
        self.assertEqual([b"\x00\x01"], charges(located, 0, b"\x00\x01\x00\x00").middle)

    def test_an_undecided_middle_and_left_neighbour_are_a_pair(self):
        cut = parity_cut([b"\x00\x01", b"\x00"])
        located = _locate(cut, WRONG_ODD_0, [0, 1, 0, 0])
        self.assertEqual(Pair(1, b"\x00"), located.outcome)
        self.assertEqual([4, 2, 1], list(located.sifts))

    def test_an_undecided_middle_and_right_neighbour_are_a_pair_at_the_middle(self):
        cut = parity_cut([b"\x00\x01", b"\x00\x01\x00"])
        located = _locate(cut, WRONG_ODD_0, [0, 1, 0, 0])
        self.assertEqual(Pair(2, b"\x00\x01"), located.outcome)
        self.assertEqual([4, 2, 1, 3], list(located.sifts))

    def test_the_search_steps_past_a_middle_whose_neighbours_read_alike(self):
        cut = parity_cut([b"\x00\x00\x00"])
        located = _locate(cut, WRONG_ODD_0, [0, 0, 0, 1, 0, 0])
        self.assertEqual(EdgeFound(ODD, 0, ODD, b"\x00\x00\x00\x01"), located.outcome)
        self.assertEqual([6, 3, 2, 4, 5], list(located.sifts))


class TestCharging(unittest.TestCase):
    def test_a_position_is_charged_to_the_walks_edge_out_of_it_else_its_last(self):
        charged = _charges(parity_cut(()), PARITY_EDGES, [1, 0])
        self.assertEqual({(ODD, 0): 2}, dict(charged.traversals))

    def test_positions_past_an_unlearned_edge_are_charged_nowhere(self):
        charged = _charges(parity_cut(()), {(EVEN, 1): (ODD, b"")}, [1, 0, 0])
        self.assertEqual({(ODD, 0): 2}, dict(charged.traversals))

    def test_an_edge_keeps_the_strings_read_undecided_at_its_positions(self):
        charged = _charges(parity_cut([b"\x01\x00"]), PARITY_EDGES, [1, 0])
        self.assertEqual({(ODD, 0): [b"\x01\x00"]}, charged.undecided_by_edge)


if __name__ == "__main__":
    unittest.main()
