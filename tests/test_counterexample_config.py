"""The shipped counterexample search config."""

import unittest

from orthogonal_dfa.l_star.counterexample_config import counterexample_search_config


class TestTheShippedConfig(unittest.TestCase):
    def test_it_derives_the_rest_from_the_family_and_the_level(self):
        config = counterexample_search_config(0.98, 0.01)
        derived = (
            config.k,
            config.max_leaves,
            config.start_rate,
            config.edge_rate,
            config.middle_rate,
            config.disagreement_rate,
            config.first_look,
            config.edge_excess,
            config.edge_traffic,
            config.search_share,
        )
        self.assertEqual(
            (0, 100, 0.05, 0.1, 0.2, 1 - 0.98, 30, 10, 0.01, 0.01), derived
        )


if __name__ == "__main__":
    unittest.main()
