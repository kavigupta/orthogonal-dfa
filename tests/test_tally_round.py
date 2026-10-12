"""The tally round against hand-built trees and scripted reads."""

import itertools
import unittest

from orthogonal_dfa.l_star.statistics import rate_side
from orthogonal_dfa.l_star.tally_round import (
    HARVEST_EDGE,
    HARVEST_MIDDLES,
    HARVEST_START,
    OUT_OF_PROBES,
    SUCCESS,
    TOO_BIG,
    Ending,
)
from tests.tally_common import (
    EVEN,
    ODD,
    PARITY_EDGES,
    WRONG_ODD_0,
    mod3_cut,
    parity_cut,
    quiet_round,
)


class TestCharging(unittest.TestCase):
    def test_counted_by_position_a_string_counts_each_time(self):
        tally = quiet_round(parity_cut([b"\x01\x00"]), PARITY_EDGES)
        tally.step(b"\x01\x00")
        tally.step(b"\x01\x00")
        self.assertEqual(4, tally.charged[(ODD, 0)])
        self.assertEqual([b"\x01\x00"] * 2, tally.edge_harvest[(ODD, 0)])

    def test_the_start_is_not_charged_to_an_edge(self):
        tally = quiet_round(parity_cut(()), PARITY_EDGES)
        tally.step(b"\x01")
        self.assertEqual({(EVEN, 1): 1}, dict(tally.charged))


class TestLearningAndRecords(unittest.TestCase):
    def test_a_member_learns_its_edge_and_starts_a_stretch(self):
        tally = quiet_round(parity_cut(()), {})
        tally.step(b"\x01\x00")
        self.assertEqual({(EVEN, 1): (ODD, b"")}, tally.edges)
        self.assertEqual((0, 1, 1), (tally.n, tally.probes, tally.version))

    def test_a_clean_disagreement_records_at_its_edge(self):
        tally = quiet_round(parity_cut(()), WRONG_ODD_0, m=3)
        tally.step(b"\x01\x00\x00\x00")
        self.assertEqual({(ODD, 0): [(b"\x01", ODD)]}, tally.records)
        self.assertEqual((1, 1), (tally.n, tally.searches))

    def test_m_records_at_another_target_redirect_the_edge(self):
        tally = quiet_round(parity_cut(()), WRONG_ODD_0)
        tally.step(b"\x01\x00\x00\x00")
        tally.step(b"\x01\x00\x01\x01")
        self.assertEqual((ODD, b"\x01"), tally.edges[(ODD, 0)])
        self.assertEqual(2, len(tally.records[(ODD, 0)]), "a redirect keeps them")
        self.assertEqual((0, 0, 1), (tally.n, tally.searches, tally.version))

    def test_one_record_short_fixes_nothing(self):
        tally = quiet_round(parity_cut(()), WRONG_ODD_0)
        tally.step(b"\x01\x00\x00\x00")
        self.assertEqual(WRONG_ODD_0, tally.edges)

    def test_m_records_at_two_targets_split_the_leaf(self):
        # The even leaf holds counts 1 and 2 mod 3; by a 1, one goes to 2 and
        # the other to 0, so the leaf splits on the 1 and the root's midfix.
        tally = quiet_round(
            mod3_cut, {(EVEN, 1): (ODD, b"\x01\x01"), (ODD, 1): (EVEN, b"")}
        )
        tally.records[(EVEN, 1)] = [(b"\x01\x01", ODD)] * 2 + [(b"\x01", EVEN)] * 2
        tally.version = 7
        self.assertIsNone(tally._settle_one())  # pylint: disable=protected-access

        self.assertEqual(b"\x01", tally.tree.midfix_at(EVEN))
        self.assertEqual({(False, False), (False, True), ODD}, tally.paths)
        self.assertEqual({}, tally.records)
        self.assertEqual(8, tally.version)
        # Out of the split leaf: dropped.  Into it: its witness re-sifted, the empty
        # string reading 0 mod 3 at the root and 1 mod 3 behind a 1.
        self.assertEqual({(ODD, 1): ((False, False), b"")}, tally.edges)

    def test_an_edge_into_the_split_leaf_whose_witness_is_undecided_is_dropped(self):
        tally = quiet_round(
            lambda z: None if z == b"\x01" else mod3_cut(z),
            {(EVEN, 1): (ODD, b"\x01\x01"), (ODD, 1): (EVEN, b"")},
        )
        tally.records[(EVEN, 1)] = [(b"\x01\x01", ODD)] * 2 + [(b"\x00", EVEN)] * 2
        tally._settle_one()  # pylint: disable=protected-access
        self.assertEqual({}, tally.edges)

    def test_a_tree_past_its_cap_ends_the_round(self):
        tally = quiet_round(mod3_cut, {(EVEN, 1): (ODD, b"")}, max_leaves=2)
        tally.records[(EVEN, 1)] = [(b"\x01\x01", ODD)] * 2 + [(b"\x01", EVEN)] * 2
        ending = tally._settle_one()  # pylint: disable=protected-access
        self.assertEqual(TOO_BIG, ending.kind)


class TestTheTests(unittest.TestCase):
    def test_rate_side(self):
        self.assertTrue(rate_side(0.1, 0.01, 1, 10, 6))
        self.assertFalse(rate_side(0.5, 0.01, 1, 10, 0))
        self.assertIsNone(rate_side(0.5, 0.01, 1, 10, 5))
        self.assertIsNone(rate_side(0.1, 0.01, 11, 10, 10), "before the first look")

    def test_an_undecided_start_harvests_it(self):
        tally = quiet_round(
            parity_cut([b""]), PARITY_EDGES, start_rate=0.05, level=1e-4
        )
        endings = [tally.step(b"\x01\x00") for _ in range(4)]
        self.assertEqual([None] * 3, endings[:3])
        self.assertEqual(Ending(HARVEST_START, [b""] * 4, None), endings[3])

    def test_an_edge_reading_undecided_past_its_excess_harvests_its_strings(self):
        tally = quiet_round(
            parity_cut([b"\x01\x00"]), PARITY_EDGES, edge_rate=0.25, edge_excess=1
        )
        self.assertIsNone(tally.step(b"\x01\x00"), "1 undecided of 2 positions")
        ending = tally.step(b"\x01\x00")
        self.assertEqual(Ending(HARVEST_EDGE, [b"\x01\x00"] * 2, (ODD, 0)), ending)

    def test_the_edge_test_waits_for_its_traffic(self):
        tally = quiet_round(
            parity_cut([b"\x01\x00"]),
            PARITY_EDGES,
            edge_rate=0.0,
            edge_excess=1,
            edge_traffic=3.0,
        )
        self.assertIsNone(tally.step(b"\x01\x00"))

    def test_undecided_middles_harvest_once_the_stretch_searches_enough(self):
        cut = parity_cut([b"\x00\x01"])
        tally = quiet_round(cut, WRONG_ODD_0, middle_rate=0.1, search_share=0.5)
        endings = [tally.step(b"\x00\x01\x00\x00") for _ in range(3)]
        self.assertEqual([None, None], endings[:2])
        self.assertEqual(Ending(HARVEST_MIDDLES, [b"\x00\x01"] * 3, None), endings[2])

    def test_the_middles_test_waits_for_searches(self):
        cut = parity_cut([b"\x00\x01"])
        tally = quiet_round(cut, WRONG_ODD_0, middle_rate=0.1, search_share=0.5)
        for _ in range(4):
            tally.step(b"\x00\x00")
        endings = [tally.step(b"\x00\x01\x00\x00") for _ in range(3)]
        self.assertEqual([None] * 3, endings, "3 searches of 7 probes")
        self.assertEqual(
            HARVEST_MIDDLES, tally.step(b"\x00\x01\x00\x00").kind, "4 of 8 probes"
        )

    def test_a_stretch_without_disagreement_succeeds(self):
        tally = quiet_round(parity_cut(()), PARITY_EDGES, disagreement_rate=0.5)
        endings = [tally.step(b"\x01\x00") for _ in range(7)]
        self.assertEqual([None] * 6, endings[:6])
        self.assertEqual(SUCCESS, endings[6].kind)

    def test_a_change_starts_the_stretch_afresh(self):
        tally = quiet_round(parity_cut(()), {**PARITY_EDGES}, disagreement_rate=0.5)
        del tally.edges[(ODD, 0)]
        endings = [tally.step(b"\x01\x01") for _ in range(6)]
        tally.step(b"\x01\x00")  # learns the edge
        endings += [tally.step(b"\x01\x01") for _ in range(6)]
        self.assertEqual([None] * 12, endings)
        self.assertEqual(SUCCESS, tally.step(b"\x01\x01").kind)
        self.assertEqual(14, tally.probes)

    def test_the_probes_running_out(self):
        tally = quiet_round(parity_cut(()), PARITY_EDGES, max_probes=3)
        self.assertEqual(OUT_OF_PROBES, tally.run(itertools.repeat(b"\x01")).kind)
        self.assertEqual(3, tally.probes)


class TestExport(unittest.TestCase):
    def test_an_unlearned_edge_loops(self):
        tally = quiet_round(parity_cut(()), {(EVEN, 1): (ODD, b"")})
        dfa = tally.to_dfa(2, 1)
        self.assertEqual({1: {0: 1, 1: 0}, 0: {0: 0, 1: 0}}, dfa.transitions)
        self.assertEqual({0}, set(dfa.final_states))


if __name__ == "__main__":
    unittest.main()
