"""Routing a string down the tree."""

import unittest

from orthogonal_dfa.l_star.midfix_tree import MidfixTree
from tests.tally_common import ODD, parity_cut


class TestRoute(unittest.TestCase):
    def test_a_decided_string_reaches_its_leaf(self):
        self.assertEqual(
            ([b"\x01"], ODD), MidfixTree([]).route(parity_cut(()), b"\x01")
        )

    def test_an_undecided_read_ends_the_route_at_its_string(self):
        tree = MidfixTree([])
        tree.split(1, b"\x07")
        reads, leaf = tree.route(parity_cut([b"\x00\x07"]), b"\x00")
        self.assertEqual(([b"\x00", b"\x00\x07"], None), (reads, leaf))


if __name__ == "__main__":
    unittest.main()
