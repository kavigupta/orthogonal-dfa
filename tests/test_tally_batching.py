"""The warm-up batching reads only what the probes it warms for read."""

import unittest

from orthogonal_dfa.l_star.tally_batching import warmed_draws
from orthogonal_dfa.l_star.tally_round import OUT_OF_PROBES, SUCCESS
from tests.tally_common import PARITY_EDGES, parity_cut, quiet_round


class TestWarming(unittest.TestCase):
    DRAWS = [bytes([i % 2, (i // 2) % 2, (i // 4) % 2, 1]) for i in range(40)]

    def test_a_held_hypothesis_warms_only_what_its_probes_read(self):
        read, warmed = set(), set()

        def cut(z):
            read.add(z)
            return parity_cut(())(z)

        stepped = quiet_round(cut, PARITY_EDGES)
        for x in self.DRAWS:
            stepped.step(x)
        draws = iter(self.DRAWS)
        tally = quiet_round(parity_cut(()), PARITY_EDGES, max_probes=len(self.DRAWS))

        ending = tally.run(warmed_draws(tally, lambda: next(draws), warmed.update))
        self.assertEqual(OUT_OF_PROBES, ending.kind)
        self.assertTrue(warmed)
        self.assertLessEqual(warmed, read)

    def test_the_block_stops_where_success_could_fire(self):
        drawn = []

        def draw():
            drawn.append(b"\x01\x00")
            return drawn[-1]

        tally = quiet_round(parity_cut(()), PARITY_EDGES, disagreement_rate=0.5)
        self.assertEqual(
            SUCCESS, tally.run(warmed_draws(tally, draw, lambda _: None)).kind
        )
        self.assertEqual(7, len(drawn))


if __name__ == "__main__":
    unittest.main()
