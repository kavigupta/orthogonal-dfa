"""The counterexample search against hand-built trees and scripted reads."""

import itertools
import unittest

from orthogonal_dfa.l_star.counterexample_probe import Agree
from orthogonal_dfa.l_star.counterexample_search import CounterexampleSearch
from orthogonal_dfa.l_star.midfix_tree import MidfixTree
from orthogonal_dfa.l_star.round_end import (
    HARVEST_EDGE,
    HARVEST_MIDDLES,
    HARVEST_START,
    OUT_OF_PROBES,
    SCORE_STALLED,
    SUCCESS,
    TOO_BIG,
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
    quiet_config,
    quiet_search,
)

#: Each probe disagrees with ``WRONG_ODD_0``, and ``MIDDLE_CUT`` leaves every
#: position between its ends undecided, so it is a middle at its first two letters,
#: one of ``MIDDLES``.
MIDDLES = [b"\x00\x01", b"\x01\x00", b"\x01\x01", b"\x00\x00"]
MIDDLE_PROBES = [
    b"\x00\x01\x00\x00",
    b"\x01\x00\x00\x00",
    b"\x01\x01\x01\x00",
    b"\x00\x00\x01\x00",
]
MIDDLE_CUT = parity_cut(
    bytes(z) for n in (1, 2, 3) for z in itertools.product((0, 1), repeat=n)
)


class TestCharging(unittest.TestCase):
    def test_counted_by_position_a_string_counts_each_time(self):
        search = quiet_search(parity_cut([b"\x01\x00"]), PARITY_EDGES)
        search.step(b"\x01\x00")
        search.step(b"\x01\x00")
        self.assertEqual(4, search.counts.traversals[(ODD, 0)])
        self.assertEqual([b"\x01\x00"] * 2, search.counts.edge_harvest[(ODD, 0)])

    def test_the_start_is_not_charged_to_an_edge(self):
        search = quiet_search(parity_cut(()), PARITY_EDGES)
        search.step(b"\x01")
        self.assertEqual({(EVEN, 1): 1}, dict(search.counts.traversals))


class TestLearningAndSplitting(unittest.TestCase):
    def test_a_member_learns_its_edge_and_starts_the_counts_afresh(self):
        search = quiet_search(parity_cut(()), {})
        search.step(b"\x01\x00")
        self.assertEqual({(EVEN, 1): (ODD, b"")}, search.hypothesis.edges)
        self.assertEqual(
            (0, {}, 1, 1),
            (
                search.counts.probes,
                dict(search.counts.traversals),
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

    def test_a_tree_past_its_cap_ends_the_round(self):
        search = quiet_search(
            mod3_cut,
            {(EVEN, 1): (ODD, b"\x01\x01"), (ODD, 1): (EVEN, b"")},
            max_leaves=2,
        )
        self.assertEqual(TOO_BIG, search.step(b"\x01\x01").kind)


class TestTheTests(unittest.TestCase):
    def test_rate_side(self):
        self.assertTrue(rate_side(0.1, 0.01, 1, 10, 6))
        self.assertFalse(rate_side(0.5, 0.01, 1, 10, 0))
        self.assertIsNone(rate_side(0.5, 0.01, 1, 10, 5))
        self.assertIsNone(rate_side(0.1, 0.01, 11, 10, 10), "before the first look")

    def test_an_undecided_start_harvests_it(self):
        starts = [bytes(bits) for bits in itertools.product((0, 1), repeat=4)][:14]
        search = quiet_search(
            parity_cut(starts),
            PARITY_EDGES,
            k=4,
            start_rate=0.05,
            start_alternative=0.1,
            level=1e-4,
        )
        # 2^n reaches 1e4 at n = 14.
        endings = [search.step(start + b"\x00") for start in starts]
        self.assertEqual([None] * 13, endings[:13])
        self.assertEqual(Ending(HARVEST_START, starts, None), endings[13])

    def test_a_start_read_again_is_not_more_evidence(self):
        search = quiet_search(
            parity_cut([b"\x01\x00\x00\x00"]),
            PARITY_EDGES,
            k=4,
            start_rate=0.05,
            start_alternative=0.1,
            level=1e-4,
        )
        endings = [search.step(b"\x01\x00\x00\x00\x00") for _ in range(14)]
        self.assertEqual([None] * 14, endings)

    def test_an_edge_reading_undecided_past_its_excess_harvests_its_strings(self):
        search = quiet_search(
            parity_cut([b"\x01\x00", b"\x00\x01\x00"]),
            PARITY_EDGES,
            edge_rate=0.25,
            edge_excess=1,
        )
        self.assertIsNone(search.step(b"\x01\x00"), "1 undecided of 1 traversal")
        ending = search.step(b"\x00\x01\x00")
        self.assertEqual(
            Ending(HARVEST_EDGE, [b"\x01\x00", b"\x00\x01\x00"], (ODD, 0)), ending
        )

    def test_an_edge_read_again_is_not_more_evidence(self):
        search = quiet_search(
            parity_cut([b"\x01\x00"]), PARITY_EDGES, edge_rate=0.25, edge_excess=1
        )
        search.step(b"\x01\x00")
        self.assertIsNone(search.step(b"\x01\x00"))

    def test_the_edge_test_waits_for_its_traffic(self):
        search = quiet_search(
            parity_cut([b"\x01\x00"]),
            PARITY_EDGES,
            edge_rate=0.0,
            edge_excess=1,
            edge_traffic=3.0,
        )
        self.assertIsNone(search.step(b"\x01\x00"))

    def test_undecided_middles_harvest_once_the_hypothesis_searches_enough(self):
        search = quiet_search(
            MIDDLE_CUT,
            WRONG_ODD_0,
            middle_rate=0.1,
            middle_alternative=0.5,
            search_share=0.5,
        )
        endings = [search.step(probe) for probe in MIDDLE_PROBES[:3]]
        self.assertEqual([None, None], endings[:2])
        self.assertEqual(Ending(HARVEST_MIDDLES, MIDDLES[:3], None), endings[2])

    def test_a_middle_read_again_is_not_more_evidence(self):
        search = quiet_search(
            MIDDLE_CUT,
            WRONG_ODD_0,
            middle_rate=0.1,
            middle_alternative=0.5,
            search_share=0.5,
        )
        endings = [search.step(MIDDLE_PROBES[0]) for _ in range(3)]
        self.assertEqual([None] * 3, endings)

    def test_the_middles_test_waits_for_searches(self):
        search = quiet_search(
            MIDDLE_CUT,
            WRONG_ODD_0,
            middle_rate=0.1,
            middle_alternative=0.5,
            search_share=0.5,
        )
        for _ in range(4):
            search.step(b"\x00\x00\x00")
        endings = [search.step(probe) for probe in MIDDLE_PROBES[:3]]
        self.assertEqual([None] * 3, endings, "3 searches of 7 probes")
        self.assertEqual(
            HARVEST_MIDDLES, search.step(MIDDLE_PROBES[3]).kind, "4 of 8 probes"
        )

    def test_a_hypothesis_without_disagreement_succeeds(self):
        search = quiet_search(parity_cut(()), PARITY_EDGES, disagreement_rate=0.5)
        endings = [search.step(b"\x01\x00") for _ in range(7)]
        self.assertEqual([None] * 6, endings[:6])
        self.assertEqual(SUCCESS, endings[6].kind)

    def test_a_change_starts_the_counts_afresh(self):
        search = quiet_search(parity_cut(()), {**PARITY_EDGES}, disagreement_rate=0.5)
        del search.hypothesis.edges[(ODD, 0)]
        endings = [search.step(b"\x01\x01") for _ in range(6)]
        search.step(b"\x01\x00")  # learns the edge
        endings += [search.step(b"\x01\x01") for _ in range(6)]
        self.assertEqual([None] * 12, endings)
        self.assertEqual(SUCCESS, search.step(b"\x01\x01").kind)
        self.assertEqual(14, search.probes)

    def test_the_probes_running_out(self):
        search = quiet_search(parity_cut(()), PARITY_EDGES, max_probes=3)
        self.assertEqual(OUT_OF_PROBES, search.run(itertools.repeat(b"\x01")).kind)
        self.assertEqual(3, search.probes)


class TestUndecidedFallbacks(unittest.TestCase):
    def search(self, cut, edges, **kwargs):
        search = CounterexampleSearch(quiet_config(), cut, MidfixTree([]), **kwargs)
        search.hypothesis.edges = dict(edges)
        return search

    def test_an_edge_its_prefix_leaves_undecided_is_read_on_a_member(self):
        edges = {e: t for e, t in PARITY_EDGES.items() if e != (ODD, 0)}
        search = self.search(
            parity_cut([b"\x01\x00"]), edges, members=[b"\x00", b"\x01\x01\x01"]
        )
        search.step(b"\x01\x00")
        self.assertEqual((ODD, b"\x01\x01\x01"), search.hypothesis.edges[(ODD, 0)])

    def test_without_a_member_the_edge_stays_unlearned(self):
        edges = {e: t for e, t in PARITY_EDGES.items() if e != (ODD, 0)}
        search = self.search(parity_cut([b"\x01\x00"]), edges, members=[b"\x00"])
        search.step(b"\x01\x00")
        self.assertNotIn((ODD, 0), search.hypothesis.edges)

    def test_closing_learns_the_edges_members_can_place(self):
        edges = {e: t for e, t in PARITY_EDGES.items() if e != (ODD, 0)}
        search = self.search(parity_cut(()), edges, members=[b"\x00", b"\x01"])
        search.close(2)
        self.assertEqual((ODD, b"\x01"), search.hypothesis.edges[(ODD, 0)])

    def test_an_undecided_empty_string_starts_where_start_says(self):
        search = self.search(parity_cut([b""]), PARITY_EDGES, start=lambda _: EVEN)
        outcome, _ = search.probe(b"\x01\x01")
        self.assertEqual(Agree(), outcome)


class TestBestHypothesis(unittest.TestCase):
    def test_the_search_ends_on_its_best_scoring_hypothesis(self):
        search = CounterexampleSearch(
            quiet_config(max_probes=4),
            parity_cut(()),
            MidfixTree([]),
            score=lambda h: -len(h.edges),
        )
        search.run(itertools.repeat(b"\x01\x00"))
        self.assertGreater(search.probes, 1)
        self.assertEqual({}, search.hypothesis.edges)

    def test_patience_ends_a_search_whose_score_stops_improving(self):
        search = CounterexampleSearch(
            quiet_config(),
            parity_cut(()),
            MidfixTree([]),
            score=lambda h: 0,
            patience=3,
        )
        search.hypothesis.edges = dict(PARITY_EDGES)
        ending = search.run(itertools.repeat(b"\x01"))
        self.assertEqual((SCORE_STALLED, 3), (ending.kind, search.probes))


class TestExport(unittest.TestCase):
    def test_an_unlearned_edge_loops(self):
        search = quiet_search(parity_cut(()), {(EVEN, 1): (ODD, b"")})
        dfa = search.hypothesis.to_dfa(2, 1)
        self.assertEqual({1: {0: 1, 1: 0}, 0: {0: 0, 1: 0}}, dfa.transitions)
        self.assertEqual({0}, set(dfa.final_states))


if __name__ == "__main__":
    unittest.main()
