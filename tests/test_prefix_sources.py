"""Sources of prefixes, one per population.

Each round hands the next one a source per population instead of the prefixes
themselves, so what a later round needs more of it can draw more of.
"""

import itertools
import unittest

import numpy as np

from orthogonal_dfa.l_star.prefix_sources import HarvestSource, Replay, harvest_reader
from orthogonal_dfa.l_star.rejection_source import SourceDry
from orthogonal_dfa.l_star.round_end import HARVEST_ENDS, HARVEST_STUCK, Ending
from tests.counterexample_common import EVEN, PARITY_EDGES, parity_cut, quiet_search


class _Finds:
    """A provenance whose every fresh draw finds ``strings``."""

    def __init__(self, *strings):
        self._strings = list(strings)

    def sample(self):
        return list(self._strings)


_FOUND = b"\x00\x01?"


class TestAHarvestSourceDrawsByProvenance(unittest.TestCase):
    def _source(self, *, known):
        return HarvestSource(
            {_Finds(_FOUND): 1},
            np.random.default_rng(0),
            known=known,
            acc_threshold=0.98,
        )

    def test_a_find_is_served(self):
        source = self._source(known=())

        self.assertTrue(source.attempt_draw())
        self.assertEqual(_FOUND, source.draw())

    def test_keeping_the_same_string_again_is_not_a_find(self):
        source = self._source(known=())

        self.assertTrue(source.attempt_draw())
        self.assertFalse(source.attempt_draw(), "the second probe found nothing new")

    def test_what_the_caller_already_holds_is_not_a_find(self):
        source = self._source(known=[_FOUND])

        self.assertFalse(source.attempt_draw())

    def test_a_source_that_finds_nothing_new_stops_rather_than_probing_forever(self):
        source = self._source(known=())

        self.assertEqual(_FOUND, source.draw())

        with self.assertRaisesRegex(SourceDry, "found no new samples"):
            source.draw()

    def test_provenances_are_drawn_in_proportion_to_what_they_found(self):
        drawn = itertools.count()

        class Finds:
            def __init__(self, tag):
                self._tag = tag

            def sample(self):
                return [next(drawn).to_bytes(4, "big") + self._tag]

        often, rarely = Finds(b"a"), Finds(b"b")
        source = HarvestSource(
            {often: 3, rarely: 1},
            np.random.default_rng(0),
            known=(),
            acc_threshold=0.98,
        )
        ends = [source.draw()[-1:] for _ in range(400)]

        self.assertAlmostEqual(ends.count(b"a") / len(ends), 3 / 4, delta=0.06)


class TestReplay(unittest.TestCase):
    def test_a_replay_reads_a_fresh_probe_where_the_round_harvested(self):
        search = quiet_search(parity_cut([b"\x01\x00"]), PARITY_EDGES)
        ending = Ending(HARVEST_ENDS, [])
        self.assertEqual(
            [b"\x01\x00"],
            Replay(search, harvest_reader(ending), lambda: b"\x01\x00").sample(),
        )
        self.assertEqual(
            [], Replay(search, harvest_reader(ending), lambda: b"\x01\x01").sample()
        )

    def test_a_stuck_harvest_reads_only_what_sticks_at_an_unlearned_edge(self):
        edges = {e: t for e, t in PARITY_EDGES.items() if e != (EVEN, 1)}
        search = quiet_search(parity_cut([b"\x00\x01", b"\x00"]), edges)
        read = harvest_reader(Ending(HARVEST_STUCK, []))
        self.assertEqual(
            [b"\x00\x01"], Replay(search, read, lambda: b"\x00\x01").sample()
        )
        self.assertEqual([], Replay(search, read, lambda: b"\x00").sample())


if __name__ == "__main__":
    unittest.main()
