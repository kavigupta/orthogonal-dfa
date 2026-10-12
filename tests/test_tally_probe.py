"""One probe against hand-built hypotheses and scripted reads."""

import unittest

from orthogonal_dfa.l_star.midfix_tree import MidfixTree
from orthogonal_dfa.l_star.tally_probe import AGREE, END, MEMBER, START, probe
from orthogonal_dfa.l_star.tally_search import EDGE, PAIR, TRIPLE
from tests.tally_common import EVEN, ODD, PARITY_EDGES, WRONG_ODD_0, parity_cut


def _probe(cut, edges, x):
    return probe(cut, MidfixTree([]), edges, 0, bytes(x))


class TestOutcomes(unittest.TestCase):
    def test_a_walk_ending_where_the_probe_sifts_agrees(self):
        self.assertEqual(
            (AGREE,), _probe(parity_cut(()), PARITY_EDGES, [1, 0, 1]).outcome
        )

    def test_an_undecided_start(self):
        result = _probe(parity_cut([b""]), PARITY_EDGES, [1, 0])
        self.assertEqual((START, b""), result.outcome)
        self.assertEqual([b""], result.start)

    def test_an_undecided_end(self):
        outcome = _probe(parity_cut([b"\x01\x00"]), PARITY_EDGES, [1, 0]).outcome
        self.assertEqual((END, b"\x01\x00"), outcome)

    def test_an_unlearned_edge_where_the_cut_places_its_prefix_is_a_member(self):
        result = _probe(parity_cut(()), {}, [1, 0])
        self.assertEqual((MEMBER, b""), result.outcome)
        self.assertEqual(((EVEN, 1), (ODD, b"")), result.learned)

    def test_an_unlearned_edge_whose_successor_is_undecided_is_an_end(self):
        outcome = _probe(parity_cut([b"\x01"]), {}, [1, 0]).outcome
        self.assertEqual((END, b"\x01"), outcome)

    def test_an_unlearned_edge_the_walk_misplaces_is_searched(self):
        edges = {(EVEN, 1): (EVEN, b"")}
        result = _probe(parity_cut(()), edges, [1, 0])
        self.assertEqual((EDGE, [EVEN, EVEN], 1), result.outcome)
        self.assertEqual([2, 1], result.sifts)

    def test_a_disagreement_is_bisected_to_its_edge(self):
        result = _probe(parity_cut(()), WRONG_ODD_0, [1, 0, 0, 0])
        self.assertEqual((EDGE, 2), (result.outcome[0], result.outcome[2]))
        self.assertEqual(((ODD, 0, ODD), b"\x01"), result.record)
        self.assertEqual([4, 2, 1], result.sifts)

    def test_an_undecided_middle_between_agreement_and_disagreement_is_a_triple(self):
        result = _probe(parity_cut([b"\x00\x01"]), WRONG_ODD_0, [0, 1, 0, 0])
        self.assertEqual((TRIPLE, 2), result.outcome)
        self.assertEqual([b"\x00\x01"], result.middle)
        self.assertIsNone(result.record)

    def test_an_undecided_middle_and_left_neighbour_are_a_pair(self):
        result = _probe(parity_cut([b"\x00\x01", b"\x00"]), WRONG_ODD_0, [0, 1, 0, 0])
        self.assertEqual((PAIR, 1), result.outcome)
        self.assertEqual([4, 2, 1], result.sifts)

    def test_an_undecided_middle_and_right_neighbour_are_a_pair_at_the_middle(self):
        cut = parity_cut([b"\x00\x01", b"\x00\x01\x00"])
        result = _probe(cut, WRONG_ODD_0, [0, 1, 0, 0])
        self.assertEqual((PAIR, 2), result.outcome)
        self.assertEqual([4, 2, 1, 3], result.sifts)

    def test_the_search_steps_past_a_middle_whose_neighbours_read_alike(self):
        result = _probe(parity_cut([b"\x00\x00\x00"]), WRONG_ODD_0, [0, 0, 0, 1, 0, 0])
        self.assertEqual((EDGE, 5), (result.outcome[0], result.outcome[2]))
        self.assertEqual(((ODD, 0, ODD), b"\x00\x00\x00\x01"), result.record)
        self.assertEqual([6, 3, 2, 4, 5], result.sifts)


class TestCharging(unittest.TestCase):
    def test_a_position_is_charged_to_the_walks_edge_out_of_it_else_its_last(self):
        result = _probe(parity_cut(()), PARITY_EDGES, [1, 0])
        self.assertEqual({(ODD, 0): 2}, dict(result.traversals))

    def test_positions_past_an_unlearned_edge_are_charged_nowhere(self):
        result = _probe(parity_cut(()), {(EVEN, 1): (ODD, b"")}, [1, 0, 0])
        self.assertEqual({(ODD, 0): 2}, dict(result.traversals))

    def test_an_edge_keeps_the_strings_read_undecided_at_its_positions(self):
        result = _probe(parity_cut([b"\x01\x00"]), PARITY_EDGES, [1, 0])
        self.assertEqual({(ODD, 0): [b"\x01\x00"]}, result.undecided_by_edge)


if __name__ == "__main__":
    unittest.main()
