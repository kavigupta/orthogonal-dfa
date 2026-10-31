import unittest

from orthogonal_dfa.l_star.counterexample_probe import (
    EdgeFound,
    Middle,
    locate,
    plain_bisect,
)
from orthogonal_dfa.l_star.fallbacks.split_gate import SplitGate
from tests.counterexample_common import EVEN, ODD, mod3_cut, quiet_search

#: The even leaf holds counts 1 and 2 mod 3, which a 1 then the root's midfix part;
#: probing two ones finds it, with the empty string's edge leading into it.
EDGES = {(EVEN, 1): (ODD, b"\x01\x01"), (ODD, 1): (EVEN, b"")}
PROBE = b"\x01\x01"
ONES = [b"\x01" * n for n in range(10)]


def undecided_at(*strings):
    return lambda z: None if z in strings else mod3_cut(z)


def gated(cut, members, reads):
    return quiet_search(cut, EDGES, split_gate=SplitGate(members, reads))


class TestSplitGate(unittest.TestCase):
    def test_a_member_read_undecided_past_the_new_node_blocks_the_split(self):
        search = gated(undecided_at(ONES[5]), [ONES[4]], 1)
        search.step(PROBE)
        self.assertEqual({EVEN, ODD}, search.hypothesis.leaves)
        self.assertEqual(
            ([ONES[5]], 1), (search.counts.middles, search.counts.searches)
        )

    def test_members_read_decided_let_the_split_through(self):
        search = gated(mod3_cut, [ONES[4], ONES[7]], 2)
        search.step(PROBE)
        self.assertEqual(3, len(search.hypothesis.leaves))

    def test_only_members_of_the_split_leaf_are_read(self):
        # Three ones sit in the other leaf.
        search = gated(undecided_at(ONES[4]), [ONES[3]], 1)
        search.step(PROBE)
        self.assertEqual(3, len(search.hypothesis.leaves))

    def test_it_reads_at_most_its_count_of_members(self):
        for reads, leaves in ((1, 3), (2, 2)):
            search = gated(undecided_at(ONES[8]), [ONES[4], ONES[7]], reads)
            search.step(PROBE)
            self.assertEqual(leaves, len(search.hypothesis.leaves), reads)

    def test_the_probes_prefix_and_the_edges_witness_are_not_read(self):
        # One one is the probe's prefix, and succeeds the edge into the leaf; two
        # ones are the edge's witness.
        search = gated(undecided_at(ONES[3], ONES[5]), [ONES[1], ONES[2], ONES[4]], 1)
        search.step(PROBE)
        self.assertEqual([ONES[5]], search.counts.middles)

    def test_the_successors_of_the_edges_into_the_leaf_are_read_first(self):
        search = quiet_search(
            undecided_at(b"\x01\x00\x01"),
            {**EDGES, (EVEN, 0): (EVEN, b"\x01")},
            split_gate=SplitGate([ONES[4]], 1),
        )
        search.step(PROBE)
        self.assertEqual([b"\x01\x00\x01"], search.counts.middles)

    def test_a_blocked_split_charges_its_reads_to_the_search(self):
        search = gated(undecided_at(ONES[8]), [ONES[4], ONES[7]], 2)
        located = locate(search.hypothesis, 0, PROBE, bisect=plain_bisect)
        self.assertIsInstance(located.outcome, EdgeFound)
        blocked = search.split_gate.read(search.hypothesis, located)
        self.assertEqual(Middle(2, ONES[8]), blocked.outcome)
        self.assertEqual(located.search_queries + 2, blocked.search_queries)


if __name__ == "__main__":
    unittest.main()
