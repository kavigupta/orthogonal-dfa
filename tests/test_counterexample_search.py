"""The counterexample search against hand-built trees and scripted reads."""

import itertools
import unittest

from orthogonal_dfa.l_star.round_end import (
    HARVEST_ENDS,
    HARVEST_MIDDLES,
    OUT_OF_PROBES,
    SUCCESS,
    Ending,
)
from orthogonal_dfa.l_star.statistics import rate_side
from tests.counterexample_common import (
    EVEN,
    ODD,
    PARITY_EDGES,
    WRONG_ODD_0,
    mod3_cut,
    parity_cut,
    quiet_search,
)


class TestCharging(unittest.TestCase):
    def test_a_string_counts_each_time_it_is_read(self):
        search = quiet_search(parity_cut([b"\x01\x00"]), PARITY_EDGES)
        search.step(b"\x01\x00")
        search.step(b"\x01\x00")
        self.assertEqual([b"\x01\x00"] * 2, search.counts.ends)
        self.assertEqual([b"\x01\x00"] * 2, search.counts.edge_harvest[(ODD, 0)])

    def test_the_start_is_not_charged_to_an_edge(self):
        search = quiet_search(parity_cut([b""]), PARITY_EDGES)
        search.step(b"\x01")
        self.assertEqual(([b""], {}), (search.counts.ends, search.counts.edge_harvest))


class TestLearningAndSplitting(unittest.TestCase):
    def test_a_member_learns_its_edge_and_starts_the_counts_afresh(self):
        search = quiet_search(parity_cut(()), {})
        search.step(b"\x01\x00")
        self.assertEqual({(EVEN, 1): (ODD, b"")}, search.hypothesis.edges)
        self.assertEqual(
            (0, 1, 1),
            (
                search.counts.probes,
                search.probes,
                search.hypothesis.version,
            ),
        )

    def test_a_disagreement_splits_the_edges_source_at_once(self):
        # The even leaf holds counts 1 and 2 mod 3; by a 1, one goes to 2 and
        # the other to 0, so the leaf splits on the 1 and the root's midfix.
        search = quiet_search(
            mod3_cut, {(EVEN, 1): (ODD, b"\x01\x01"), (ODD, 1): (EVEN, b"")}
        )
        search.hypothesis.version = 7
        self.assertIsNone(search.step(b"\x01\x01"))

        hypothesis = search.hypothesis
        self.assertEqual(b"\x01", hypothesis.tree.midfix_at(EVEN))
        self.assertEqual({(False, False), (False, True), ODD}, hypothesis.leaves)
        self.assertEqual((8, 0), (hypothesis.version, search.counts.probes))
        # Out of the split leaf: dropped.  Into it: its witness re-sifted, the empty
        # string reading 0 mod 3 at the root and 1 mod 3 behind a 1.
        self.assertEqual({(ODD, 1): ((False, False), b"")}, hypothesis.edges)

    def test_an_edge_into_the_split_leaf_whose_witness_is_undecided_is_dropped(self):
        search = quiet_search(
            lambda z: None if z == b"\x01" else mod3_cut(z),
            {
                (EVEN, 1): (ODD, b"\x01\x01"),
                (ODD, 1): (EVEN, b""),
                (ODD, 0): (ODD, b""),
            },
        )
        search.step(b"\x00\x01\x01")
        self.assertEqual({(ODD, 0): (ODD, b"")}, search.hypothesis.edges)


class TestTheTests(unittest.TestCase):
    def test_rate_side(self):
        self.assertTrue(rate_side(0.1, 0.01, 1, 10, 6))
        self.assertFalse(rate_side(0.5, 0.01, 1, 10, 0))
        self.assertIsNone(rate_side(0.5, 0.01, 1, 10, 5))
        self.assertIsNone(rate_side(0.1, 0.01, 11, 10, 10), "before the first look")

    def test_an_undecided_start_harvests_it(self):
        search = quiet_search(
            parity_cut([b""]),
            PARITY_EDGES,
            ends_rate=0.05,
            ends_alternative=0.1,
            level=1e-4,
        )
        # 2^n reaches 1e4 at n = 14.
        endings = [search.step(b"\x01\x00") for _ in range(14)]
        self.assertEqual([None] * 13, endings[:13])
        self.assertEqual(Ending(HARVEST_ENDS, [b""] * 14), endings[13])

    def test_an_undecided_end_harvests_it(self):
        search = quiet_search(
            parity_cut([b"\x01\x00"]),
            PARITY_EDGES,
            ends_rate=0.05,
            ends_alternative=0.1,
            level=1e-4,
        )
        endings = [search.step(b"\x01\x00") for _ in range(14)]
        self.assertEqual([None] * 13, endings[:13])
        self.assertEqual(Ending(HARVEST_ENDS, [b"\x01\x00"] * 14), endings[13])

    def test_undecided_middles_harvest_once_the_hypothesis_searches_enough(self):
        cut = parity_cut([b"\x00\x01"])
        search = quiet_search(
            cut, WRONG_ODD_0, middle_rate=0.1, middle_alternative=0.5, search_share=0.5
        )
        endings = [search.step(b"\x00\x01\x00\x00") for _ in range(3)]
        self.assertEqual([None, None], endings[:2])
        self.assertEqual(Ending(HARVEST_MIDDLES, [b"\x00\x01"] * 3), endings[2])

    def test_the_middles_test_waits_for_searches(self):
        cut = parity_cut([b"\x00\x01"])
        search = quiet_search(
            cut, WRONG_ODD_0, middle_rate=0.1, middle_alternative=0.5, search_share=0.5
        )
        for _ in range(4):
            search.step(b"\x00\x00")
        endings = [search.step(b"\x00\x01\x00\x00") for _ in range(3)]
        self.assertEqual([None] * 3, endings, "3 searches of 7 probes")
        self.assertEqual(
            HARVEST_MIDDLES, search.step(b"\x00\x01\x00\x00").kind, "4 of 8 probes"
        )

    def test_a_hypothesis_without_disagreement_succeeds(self):
        search = quiet_search(parity_cut(()), PARITY_EDGES, disagreement_rate=0.5)
        endings = [search.step(b"\x01\x00") for _ in range(7)]
        self.assertEqual([None] * 6, endings[:6])
        self.assertEqual(SUCCESS, endings[6].kind)

    def test_an_undecided_end_delays_success(self):
        def probes_to_success(undecided):
            search = quiet_search(
                parity_cut(undecided),
                PARITY_EDGES,
                disagreement_rate=0.5,
                disagreement_alternative=0.1,
                level=0.01,
            )
            search.step(b"\x01\x01\x01")
            while search.step(b"\x01\x00") is None:
                pass
            return search.probes

        self.assertEqual(8, probes_to_success([]))
        self.assertEqual(12, probes_to_success([b"\x01\x01\x01"]))

    def test_a_change_starts_the_counts_afresh(self):
        search = quiet_search(parity_cut(()), {**PARITY_EDGES}, disagreement_rate=0.5)
        del search.hypothesis.edges[(ODD, 0)]
        endings = [search.step(b"\x01\x01") for _ in range(6)]
        search.step(b"\x01\x00")  # learns the edge
        endings += [search.step(b"\x01\x01") for _ in range(6)]
        self.assertEqual([None] * 12, endings)
        self.assertEqual(SUCCESS, search.step(b"\x01\x01").kind)
        self.assertEqual(14, search.probes)

    def test_undecided_ends_count_against_success(self):
        # Ends at the shipped config's cap on a deep tree, where they never harvest.
        search = quiet_search(
            parity_cut([b"\x00\x01"]),
            {e: t for e, t in PARITY_EDGES.items() if e != (EVEN, 1)},
            ends_rate=0.5,
            ends_alternative=1.0,
            disagreement_rate=0.02,
            disagreement_alternative=0.002,
            level=1e-3,
            max_probes=1000,
        )
        ending = search.run(itertools.cycle([b"\x00\x01", b"\x00", b"\x00\x00"]))
        self.assertNotEqual(
            SUCCESS,
            ending.kind,
            f"{len(search.counts.ends)} of {search.counts.probes} probes ended undecided",
        )

    def test_the_probes_running_out(self):
        search = quiet_search(parity_cut(()), PARITY_EDGES, max_probes=3)
        self.assertEqual(OUT_OF_PROBES, search.run(itertools.repeat(b"\x01")).kind)
        self.assertEqual(3, search.probes)


class TestExport(unittest.TestCase):
    def test_an_unlearned_edge_loops(self):
        search = quiet_search(parity_cut(()), {(EVEN, 1): (ODD, b"")})
        dfa = search.hypothesis.to_dfa(2, 1)
        self.assertEqual({1: {0: 1, 1: 0}, 0: {0: 0, 1: 0}}, dfa.transitions)
        self.assertEqual({0}, set(dfa.final_states))


if __name__ == "__main__":
    unittest.main()
