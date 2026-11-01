import unittest

import numpy as np
from permacache import no_cache_global

from orthogonal_dfa.experiments.length_oracles.oracles import PerLengthOracle, oracle
from orthogonal_dfa.l_star.examples.spliceai_oracle import median_threshold
from tests.spliceai_small import (
    QUERY_LENGTH,
    random_middles,
    small_module_and_exon,
    small_score_model_and_exon,
)

LENGTHS = range(QUERY_LENGTH - 2, QUERY_LENGTH + 3)


class TestPerLengthOracle(unittest.TestCase):
    def setUp(self):
        self.score_model, self.exon = small_score_model_and_exon()

    def _oracle(self, thresholds):
        return PerLengthOracle(
            self.exon, self.score_model, thresholds, device="cpu", chunk=64
        )

    def _accepts(self, oracle_, length, seed):
        middles, _ = random_middles(256, length, seed)
        return oracle_.membership_queries(middles)

    def test_accepts_about_half_at_each_length(self):
        with no_cache_global():
            thresholds = {
                l: median_threshold(
                    self.score_model, self.exon, l, count=512, device="cpu"
                )
                for l in LENGTHS
            }
        per_length = self._oracle(thresholds)
        for l in LENGTHS:
            self.assertAlmostEqual(
                self._accepts(per_length, l, seed=l).mean(), 0.5, delta=0.2
            )

    def test_reads_each_lengths_own_threshold(self):
        thresholds = {QUERY_LENGTH: -np.inf, QUERY_LENGTH + 1: np.inf}
        per_length = self._oracle(thresholds)
        self.assertEqual(per_length.thresholds, thresholds)
        self.assertTrue(self._accepts(per_length, QUERY_LENGTH, seed=0).all())
        self.assertFalse(self._accepts(per_length, QUERY_LENGTH + 1, seed=0).any())

    def test_mixed_lengths_in_one_query(self):
        per_length = self._oracle({QUERY_LENGTH: -np.inf, QUERY_LENGTH + 1: np.inf})
        low, _ = random_middles(4, QUERY_LENGTH, 0)
        high, _ = random_middles(4, QUERY_LENGTH + 1, 0)
        np.testing.assert_array_equal(
            per_length.membership_queries(low + high),
            [True] * 4 + [False] * 4,
        )

    def test_lengths_outside_the_band_take_the_nearest_threshold(self):
        per_length = self._oracle({QUERY_LENGTH: -np.inf, QUERY_LENGTH + 1: np.inf})
        self.assertTrue(self._accepts(per_length, QUERY_LENGTH - 5, seed=0).all())
        self.assertFalse(self._accepts(per_length, QUERY_LENGTH + 6, seed=0).any())


class TestOracle(unittest.TestCase):
    def test_unknown_kind_is_rejected_before_fitting(self):
        module, exon = small_module_and_exon()
        with self.assertRaises(AssertionError):
            oracle("sloped", exon, module, device="cpu")


if __name__ == "__main__":
    unittest.main()
