"""The shipped counterexample search config."""

import unittest

from orthogonal_dfa.l_star.counterexample_config import counterexample_search_config


class TestTheShippedConfig(unittest.TestCase):
    def test_it_derives_the_rest_from_the_family_and_the_level(self):
        config = counterexample_search_config(0.98, 0.01)
        derived = (
            config.k,
            config.ends_rate(2),
            config.ends_rate(9),
            config.middle_rate(4),
            config.middle_rate(30),
            config.disagreement_rate,
            config.ends_alternative(2),
            config.middle_alternative(4),
            config.disagreement_alternative,
            config.search_share,
        )
        self.assertEqual(
            (
                0,
                0.2,
                0.5,
                0.2,
                1.0,
                1 - 0.98,
                0.4,
                0.25,
                0.1 * (1 - 0.98),
                0.01,
            ),
            derived,
        )


if __name__ == "__main__":
    unittest.main()
