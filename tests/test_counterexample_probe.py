"""One probe against hand-built hypotheses and scripted reads."""

import unittest

from orthogonal_dfa.l_star.counterexample_probe import (
    Agree,
    EdgeFound,
    End,
    Member,
    Middle,
    Start,
    charges,
    locate,
)
from orthogonal_dfa.l_star.hypothesis_dfa import HypothesisDFA
from orthogonal_dfa.l_star.midfix_tree import MidfixTree
from tests.counterexample_common import EVEN, ODD, PARITY_EDGES, WRONG_ODD_0, parity_cut


def _locate(cut, edges, x):
    hypothesis = HypothesisDFA(cut, MidfixTree([]))
    hypothesis.edges = dict(edges)
    return locate(hypothesis, 0, bytes(x))


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
        self.assertEqual([b""], _charges(parity_cut([b""]), PARITY_EDGES, [1, 0]).ends)

    def test_an_undecided_end(self):
        located = _locate(parity_cut([b"\x01\x00"]), PARITY_EDGES, [1, 0])
        self.assertEqual(End(b"\x01\x00"), located.outcome)
        self.assertEqual([b"\x01\x00"], charges(located, 0, b"\x01\x00").ends)

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

    def test_an_undecided_midpoint_ends_the_search_as_a_middle(self):
        located = _locate(parity_cut([b"\x00\x01"]), WRONG_ODD_0, [0, 1, 0, 0])
        self.assertEqual(Middle(2, b"\x00\x01"), located.outcome)
        self.assertEqual([4, 2], list(located.sifts))
        self.assertEqual([b"\x00\x01"], charges(located, 0, b"\x00\x01\x00\x00").middle)

    def test_the_search_does_not_step_past_an_undecided_midpoint(self):
        cut = parity_cut([b"\x00\x00\x00"])
        located = _locate(cut, WRONG_ODD_0, [0, 0, 0, 1, 0, 0])
        self.assertEqual(Middle(3, b"\x00\x00\x00"), located.outcome)
        self.assertEqual([6, 3], list(located.sifts))


class TestCharging(unittest.TestCase):
    def test_an_edge_keeps_the_strings_read_undecided_at_its_positions(self):
        charged = _charges(parity_cut([b"\x01\x00"]), PARITY_EDGES, [1, 0])
        self.assertEqual({(ODD, 0): [b"\x01\x00"]}, charged.undecided_by_edge)


if __name__ == "__main__":
    unittest.main()
