import unittest

import numpy as np

from orthogonal_dfa.analysis.prefix_information import (
    independence_floor,
    prefix_explained_variance,
)


class TestPrefixExplainedVariance(unittest.TestCase):
    def test_constant_rows_explain_everything(self):
        # each prefix has a fixed accept pattern across suffixes -> prefix explains all.
        rows = np.random.default_rng(0).integers(0, 2, size=(20, 1))
        matrix = np.tile(rows, (1, 50)).astype(bool)
        self.assertAlmostEqual(prefix_explained_variance(matrix), 1.0)

    def test_iid_matrix_sits_near_the_one_over_k_floor(self):
        k = 200
        matrix = np.random.default_rng(1).integers(0, 2, size=(300, k)).astype(bool)
        self.assertLess(prefix_explained_variance(matrix), 3 / k)

    def test_constant_matrix_has_no_variance(self):
        self.assertEqual(prefix_explained_variance(np.ones((5, 5), bool)), 0.0)

    def test_partial_structure_is_between_floor_and_one(self):
        rng = np.random.default_rng(2)
        k = 100
        propensity = rng.random((100, 1))  # per-prefix accept tendency
        matrix = (propensity + rng.random((100, k))) / 2 > 0.5
        pev = prefix_explained_variance(matrix)
        self.assertGreater(pev, 5 / k)
        self.assertLess(pev, 1.0)


class TestIndependenceFloor(unittest.TestCase):
    def test_floor_destroys_prefix_structure(self):
        # a perfectly prefix-structured matrix: explained ~1, but the floor ~1/k.
        rows = (np.arange(40) % 2).reshape(-1, 1)
        matrix = np.tile(rows, (1, 128)).astype(bool)
        self.assertGreater(prefix_explained_variance(matrix), 0.9)
        self.assertLess(independence_floor(matrix), 0.05)


if __name__ == "__main__":
    unittest.main()
