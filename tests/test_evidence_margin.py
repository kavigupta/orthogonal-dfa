import unittest

from orthogonal_dfa.l_star.statistics import population_size_and_evidence_margin


class TestMinMargin(unittest.TestCase):
    def test_a_margin_no_cross_limit_above_1e_30_buys_is_reached(self):
        # max_coverage_error = 1/12 at signal 0.3 asks for 0.25, which takes a cross
        # rate near 1e-123.
        _, eps = population_size_and_evidence_margin(
            0.3, 1.5e-7, 0.01, center=0.65, min_margin=0.25
        )
        self.assertGreaterEqual(eps, 0.25)

    def test_each_halving_of_the_coverage_error_quadruples_the_family(self):
        sizes = [
            population_size_and_evidence_margin(
                0.3, 1.5e-7, 0.01, center=0.65, min_margin=0.3 * (1 - 2 * ecov)
            )[0]
            for ecov in (1 / 12, 1 / 24)
        ]
        self.assertAlmostEqual(sizes[1] / sizes[0], 4, delta=0.5)


if __name__ == "__main__":
    unittest.main()
