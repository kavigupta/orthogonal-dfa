"""The tally round against hand-built trees and scripted reads."""

import unittest
from dataclasses import replace

from orthogonal_dfa.l_star.midfix_tree import MidfixTree
from orthogonal_dfa.l_star.tally_round import (
    AGREE,
    EDGE,
    END,
    HARVEST_EDGE,
    HARVEST_MIDDLES,
    HARVEST_START,
    MEMBER,
    OUT_OF_PROBES,
    PAIR,
    START,
    SUCCESS,
    TOO_BIG,
    TRIPLE,
    Ending,
    Probe,
    Replay,
    TallyConfig,
    TallyRound,
    rate_side,
    route,
)

EVEN, ODD = (False,), (True,)

#: Parity of the ones, which the root's cut alone reads.
PARITY_EDGES = {
    (EVEN, 0): (EVEN, b""),
    (EVEN, 1): (ODD, b""),
    (ODD, 0): (ODD, b"\x01"),
    (ODD, 1): (EVEN, b"\x01"),
}

#: Parity with the edge out of an odd leaf by a 0 pointing the wrong way.
WRONG_ODD_0 = {**PARITY_EDGES, (ODD, 0): (EVEN, b"\x01")}


def _cut(undecided=()):
    """Accepts an odd count of ones, and leaves ``undecided`` undecided."""
    undecided = set(undecided)

    def cut(z):
        return None if z in undecided else sum(z) % 2 == 1

    return cut


def _mod3_cut(z):
    """Accepts a count of ones divisible by 3: the root's leaves conflate 1 and 2."""
    return sum(z) % 3 == 0


def _config(**changes):
    """A config whose tests never fire and whose fixes need two records."""
    quiet = TallyConfig(
        k=0,
        m=2,
        max_leaves=10,
        start_rate=1.0,
        edge_rate=0.0,
        middle_rate=1.0,
        disagreement_rate=0.0,
        level=0.01,
        first_look=1,
        edge_excess=1e9,
        edge_traffic=0.0,
        search_share=1e9,
        max_probes=100,
        count_reads=False,
    )
    return replace(quiet, **changes)


def _round(cut, edges, **changes):
    tally = TallyRound(_config(**changes), cut, MidfixTree([]))
    tally.edges = dict(edges)
    return tally


def _probe(cut, edges, x):
    return Probe(cut, MidfixTree([]).root, edges, k=0, x=bytes(x))


class TestRoute(unittest.TestCase):
    def test_a_decided_string_reaches_its_leaf(self):
        self.assertEqual(([b"\x01"], ODD), route(_cut(), MidfixTree([]).root, b"\x01"))

    def test_an_undecided_read_ends_the_route_at_its_string(self):
        tree = MidfixTree([])
        tree.split(1, b"\x07")
        reads, leaf = route(_cut([b"\x00\x07"]), tree.root, b"\x00")
        self.assertEqual(([b"\x00", b"\x00\x07"], None), (reads, leaf))


class TestOutcomes(unittest.TestCase):
    def test_a_walk_ending_where_the_probe_sifts_agrees(self):
        self.assertEqual((AGREE,), _probe(_cut(), PARITY_EDGES, [1, 0, 1]).outcome())

    def test_an_undecided_start(self):
        probe = _probe(_cut([b""]), PARITY_EDGES, [1, 0])
        self.assertEqual((START, b""), probe.outcome())
        self.assertEqual([b""], probe.start())

    def test_an_undecided_end(self):
        outcome = _probe(_cut([b"\x01\x00"]), PARITY_EDGES, [1, 0]).outcome()
        self.assertEqual((END, b"\x01\x00"), outcome)

    def test_an_unlearned_edge_where_the_cut_places_its_prefix_is_a_member(self):
        self.assertEqual((MEMBER, b""), _probe(_cut(), {}, [1, 0]).outcome())

    def test_an_unlearned_edge_whose_successor_is_undecided_is_an_end(self):
        self.assertEqual((END, b"\x01"), _probe(_cut([b"\x01"]), {}, [1, 0]).outcome())

    def test_an_unlearned_edge_the_walk_misplaces_is_searched(self):
        edges = {(EVEN, 1): (EVEN, b"")}
        probe = _probe(_cut(), edges, [1, 0])
        self.assertEqual((EDGE, [EVEN, EVEN], 1), probe.outcome())
        self.assertEqual([2, 1], probe.sifts())

    def test_a_disagreement_is_bisected_to_its_edge(self):
        probe = _probe(_cut(), WRONG_ODD_0, [1, 0, 0, 0])
        outcome = probe.outcome()
        self.assertEqual((EDGE, 2), (outcome[0], outcome[2]))
        self.assertEqual(((ODD, 0, ODD), b"\x01"), probe.record(outcome))
        self.assertEqual([4, 2, 1], probe.sifts())

    def test_an_undecided_middle_between_agreement_and_disagreement_is_a_triple(self):
        probe = _probe(_cut([b"\x00\x01"]), WRONG_ODD_0, [0, 1, 0, 0])
        self.assertEqual((TRIPLE, 2), probe.outcome())
        self.assertEqual([b"\x00\x01"], probe.middle(probe.outcome()))
        self.assertIsNone(probe.record(probe.outcome()))

    def test_an_undecided_middle_and_left_neighbour_are_a_pair(self):
        probe = _probe(_cut([b"\x00\x01", b"\x00"]), WRONG_ODD_0, [0, 1, 0, 0])
        self.assertEqual((PAIR, 1), probe.outcome())
        self.assertEqual([4, 2, 1], probe.sifts())

    def test_an_undecided_middle_and_right_neighbour_are_a_pair_at_the_middle(self):
        probe = _probe(_cut([b"\x00\x01", b"\x00\x01\x00"]), WRONG_ODD_0, [0, 1, 0, 0])
        self.assertEqual((PAIR, 2), probe.outcome())
        self.assertEqual([4, 2, 1, 3], probe.sifts())

    def test_the_search_steps_past_a_middle_whose_neighbours_read_alike(self):
        probe = _probe(_cut([b"\x00\x00\x00"]), WRONG_ODD_0, [0, 0, 0, 1, 0, 0])
        outcome = probe.outcome()
        self.assertEqual((EDGE, 5), (outcome[0], outcome[2]))
        self.assertEqual(((ODD, 0, ODD), b"\x00\x00\x00\x01"), probe.record(outcome))
        self.assertEqual([6, 3, 2, 4, 5], probe.sifts())


class TestCharging(unittest.TestCase):
    def test_a_position_is_charged_to_the_walks_edge_out_of_it_else_its_last(self):
        probe = _probe(_cut(), PARITY_EDGES, [1, 0])
        self.assertEqual((EVEN, 1), probe.pos_edge(0))
        self.assertEqual((ODD, 0), probe.pos_edge(1))
        self.assertEqual((ODD, 0), probe.pos_edge(2))
        self.assertEqual({(ODD, 0): 2}, dict(probe.traversals()))

    def test_positions_past_an_unlearned_edge_are_charged_nowhere(self):
        probe = _probe(_cut(), {(EVEN, 1): (ODD, b"")}, [1, 0, 0])
        self.assertIsNone(probe.pos_edge(3))
        self.assertEqual({(ODD, 0): 2}, dict(probe.traversals()))

    def test_an_edge_keeps_the_strings_read_undecided_at_its_positions(self):
        probe = _probe(_cut([b"\x01\x00"]), PARITY_EDGES, [1, 0])
        self.assertEqual([b"\x01\x00"], probe.edge_harvest((ODD, 0)))
        self.assertEqual([], probe.edge_harvest((EVEN, 1)))

    def test_counted_by_position_a_string_counts_each_time(self):
        tally = _round(_cut([b"\x01\x00"]), PARITY_EDGES)
        tally.step(b"\x01\x00")
        tally.step(b"\x01\x00")
        self.assertEqual(4, tally.charged[(ODD, 0)])
        self.assertEqual([b"\x01\x00"] * 2, tally.edge_harvest[(ODD, 0)])

    def test_counted_by_read_a_string_counts_at_its_first_read_only(self):
        tally = _round(_cut([b"\x01\x00"]), PARITY_EDGES, count_reads=True)
        tally.step(b"\x01\x00")
        tally.step(b"\x01\x00")
        self.assertEqual(1, tally.charged[(ODD, 0)])
        self.assertEqual([b"\x01\x00"], tally.edge_harvest[(ODD, 0)])

    def test_the_start_is_not_charged_to_an_edge(self):
        tally = _round(_cut(), PARITY_EDGES, count_reads=True)
        tally.step(b"\x01")
        self.assertEqual({(EVEN, 1): 1}, dict(tally.charged))


class TestLearningAndRecords(unittest.TestCase):
    def test_a_member_learns_its_edge_and_starts_a_stretch(self):
        tally = _round(_cut(), {})
        tally.step(b"\x01\x00")
        self.assertEqual({(EVEN, 1): (ODD, b"")}, tally.edges)
        self.assertEqual((0, 1, 1), (tally.n, tally.probes, tally.version))

    def test_a_clean_disagreement_records_at_its_edge(self):
        tally = _round(_cut(), WRONG_ODD_0, m=3)
        tally.step(b"\x01\x00\x00\x00")
        self.assertEqual({(ODD, 0): [(b"\x01", ODD)]}, tally.records)
        self.assertEqual((1, 1), (tally.n, tally.searches))

    def test_m_records_at_another_target_redirect_the_edge(self):
        tally = _round(_cut(), WRONG_ODD_0)
        tally.step(b"\x01\x00\x00\x00")
        tally.step(b"\x01\x00\x01\x01")
        self.assertEqual((ODD, b"\x01"), tally.edges[(ODD, 0)])
        self.assertEqual(2, len(tally.records[(ODD, 0)]), "a redirect keeps them")
        self.assertEqual((0, 0, 1), (tally.n, tally.searches, tally.version))

    def test_one_record_short_fixes_nothing(self):
        tally = _round(_cut(), WRONG_ODD_0)
        tally.step(b"\x01\x00\x00\x00")
        self.assertEqual(WRONG_ODD_0, tally.edges)

    def test_m_records_at_two_targets_split_the_leaf(self):
        # The even leaf holds counts 1 and 2 mod 3; by a 1, one goes to 2 and
        # the other to 0, so the leaf splits on the 1 and the root's midfix.
        tally = _round(
            _mod3_cut, {(EVEN, 1): (ODD, b"\x01\x01"), (ODD, 1): (EVEN, b"")}
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
        tally = _round(
            lambda z: None if z == b"\x01" else _mod3_cut(z),
            {(EVEN, 1): (ODD, b"\x01\x01"), (ODD, 1): (EVEN, b"")},
        )
        tally.records[(EVEN, 1)] = [(b"\x01\x01", ODD)] * 2 + [(b"\x00", EVEN)] * 2
        tally._settle_one()  # pylint: disable=protected-access
        self.assertEqual({}, tally.edges)

    def test_a_tree_past_its_cap_ends_the_round(self):
        tally = _round(_mod3_cut, {(EVEN, 1): (ODD, b"")}, max_leaves=2)
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
        tally = _round(_cut([b""]), PARITY_EDGES, start_rate=0.05, level=1e-4)
        endings = [tally.step(b"\x01\x00") for _ in range(4)]
        self.assertEqual([None] * 3, endings[:3])
        self.assertEqual(Ending(HARVEST_START, [b""] * 4, None), endings[3])

    def test_an_edge_reading_undecided_past_its_excess_harvests_its_strings(self):
        tally = _round(_cut([b"\x01\x00"]), PARITY_EDGES, edge_rate=0.25, edge_excess=1)
        self.assertIsNone(tally.step(b"\x01\x00"), "1 undecided of 2 positions")
        ending = tally.step(b"\x01\x00")
        self.assertEqual(Ending(HARVEST_EDGE, [b"\x01\x00"] * 2, (ODD, 0)), ending)

    def test_the_edge_test_waits_for_its_traffic(self):
        tally = _round(
            _cut([b"\x01\x00"]),
            PARITY_EDGES,
            edge_rate=0.0,
            edge_excess=1,
            edge_traffic=3.0,
        )
        self.assertIsNone(tally.step(b"\x01\x00"))

    def test_undecided_middles_harvest_once_the_stretch_searches_enough(self):
        cut = _cut([b"\x00\x01"])
        tally = _round(cut, WRONG_ODD_0, middle_rate=0.1, search_share=0.5)
        endings = [tally.step(b"\x00\x01\x00\x00") for _ in range(3)]
        self.assertEqual([None, None], endings[:2])
        self.assertEqual(Ending(HARVEST_MIDDLES, [b"\x00\x01"] * 3, None), endings[2])

    def test_the_middles_test_waits_for_searches(self):
        cut = _cut([b"\x00\x01"])
        tally = _round(cut, WRONG_ODD_0, middle_rate=0.1, search_share=0.5)
        for _ in range(4):
            tally.step(b"\x00\x00")
        endings = [tally.step(b"\x00\x01\x00\x00") for _ in range(3)]
        self.assertEqual([None] * 3, endings, "3 searches of 7 probes")
        self.assertEqual(
            HARVEST_MIDDLES, tally.step(b"\x00\x01\x00\x00").kind, "4 of 8 probes"
        )

    def test_a_stretch_without_disagreement_succeeds(self):
        tally = _round(_cut(), PARITY_EDGES, disagreement_rate=0.5)
        endings = [tally.step(b"\x01\x00") for _ in range(7)]
        self.assertEqual([None] * 6, endings[:6])
        self.assertEqual(SUCCESS, endings[6].kind)

    def test_a_change_starts_the_stretch_afresh(self):
        tally = _round(_cut(), {**PARITY_EDGES}, disagreement_rate=0.5)
        del tally.edges[(ODD, 0)]
        endings = [tally.step(b"\x01\x01") for _ in range(6)]
        tally.step(b"\x01\x00")  # learns the edge
        endings += [tally.step(b"\x01\x01") for _ in range(6)]
        self.assertEqual([None] * 12, endings)
        self.assertEqual(SUCCESS, tally.step(b"\x01\x01").kind)
        self.assertEqual(14, tally.probes)

    def test_the_probes_running_out(self):
        tally = _round(_cut(), PARITY_EDGES, max_probes=3)
        self.assertEqual(OUT_OF_PROBES, tally.run(lambda: b"\x01").kind)
        self.assertEqual(3, tally.probes)


class TestExport(unittest.TestCase):
    def test_an_unlearned_edge_loops(self):
        tally = _round(_cut(), {(EVEN, 1): (ODD, b"")})
        dfa = tally.to_dfa(2, 1)
        self.assertEqual({1: {0: 1, 1: 0}, 0: {0: 0, 1: 0}}, dfa.transitions)
        self.assertEqual({0}, set(dfa.final_states))


class TestReplay(unittest.TestCase):
    def test_a_replay_reads_a_fresh_probe_where_the_round_harvested(self):
        tally = _round(_cut([b"\x01\x00"]), PARITY_EDGES)
        ending = Ending(HARVEST_EDGE, [], (ODD, 0))
        self.assertEqual(
            [b"\x01\x00"], Replay(tally, ending, lambda: b"\x01\x00").sample()
        )
        self.assertEqual([], Replay(tally, ending, lambda: b"\x01\x01").sample())


if __name__ == "__main__":
    unittest.main()
