"""Skipping search only ever reports an edge between adjacent decided positions."""

import itertools
import unittest

from orthogonal_dfa.l_star.fallbacks.skipping import skipping_search
from orthogonal_dfa.l_star.tripartite_search import EDGE, UNDECIDED_MID


class TestSkippingSearch(unittest.TestCase):
    def test_every_edge_it_reports_is_between_adjacent_decided_positions(self):
        for n in range(2, 8):
            for middle in itertools.product((True, False, None), repeat=n - 2):
                table = (True,) + middle + (False,)
                found = skipping_search(table.__getitem__, 0, n - 1)
                if found.kind == EDGE:
                    j = found.position
                    self.assertEqual((True, False), table[j - 1 : j + 1], table)
                else:
                    self.assertEqual(UNDECIDED_MID, found.kind, table)

    def test_an_undecided_midpoint_is_stepped_past(self):
        table = (True, True, None, True, False)
        found = skipping_search(lambda i: table[i], 0, 4)
        self.assertEqual((EDGE, 4), (found.kind, found.position))

    def test_only_undecided_between_the_bounds_is_a_middle(self):
        table = (True, None, None, False)
        found = skipping_search(lambda i: table[i], 0, 3)
        self.assertEqual(UNDECIDED_MID, found.kind)
