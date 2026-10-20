"""The warm-up batching reads only what the probes it warms for read."""

import unittest

from orthogonal_dfa.l_star.probe_batching import warmed_draws
from orthogonal_dfa.l_star.round_end import OUT_OF_PROBES, SUCCESS
from tests.counterexample_common import EVEN, PARITY_EDGES, parity_cut, quiet_search


class TestWarming(unittest.TestCase):
    DRAWS = [bytes([i % 2, (i // 2) % 2, (i // 4) % 2, 1]) for i in range(40)]

    def test_a_held_hypothesis_warms_only_what_its_probes_read(self):
        read, warmed = set(), set()

        def cut(z):
            read.add(z)
            return parity_cut(())(z)

        stepped = quiet_search(cut, PARITY_EDGES)
        for x in self.DRAWS:
            stepped.step(x)
        draws = iter(self.DRAWS)
        search = quiet_search(parity_cut(()), PARITY_EDGES, max_probes=len(self.DRAWS))

        ending = search.run(warmed_draws(search, lambda: next(draws), warmed.update))
        self.assertEqual(OUT_OF_PROBES, ending.kind)
        self.assertTrue(warmed)
        self.assertLessEqual(warmed, read)

    def test_the_start_sifts_are_warmed(self):
        warmed = set()
        draws = iter(self.DRAWS)
        search = quiet_search(
            parity_cut(()), PARITY_EDGES, k=2, max_probes=len(self.DRAWS)
        )
        search.run(warmed_draws(search, lambda: next(draws), warmed.update))
        self.assertLessEqual({x[:2] for x in self.DRAWS}, warmed)

    def test_the_block_stops_where_success_could_fire(self):
        drawn = []

        def draw():
            drawn.append(b"\x01\x00")
            return drawn[-1]

        search = quiet_search(parity_cut(()), PARITY_EDGES, disagreement_rate=0.5)
        self.assertEqual(
            SUCCESS, search.run(warmed_draws(search, draw, lambda _: None)).kind
        )
        self.assertEqual(7, len(drawn))

    def test_the_block_counts_a_stuck_end_against_success(self):
        drawn = []

        def draw():
            drawn.append(b"\x00" if drawn else b"\x00\x01")
            return drawn[-1]

        edges = {e: t for e, t in PARITY_EDGES.items() if e != (EVEN, 1)}
        search = quiet_search(
            parity_cut([b"\x00\x01"]),
            edges,
            disagreement_rate=0.5,
            disagreement_alternative=0.05,
        )
        self.assertEqual(
            SUCCESS, search.run(warmed_draws(search, draw, lambda _: None)).kind
        )
        self.assertEqual((search.probes, 1), (len(drawn), len(search.counts.stuck)))


if __name__ == "__main__":
    unittest.main()
