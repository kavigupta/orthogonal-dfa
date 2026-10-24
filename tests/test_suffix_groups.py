import unittest
from unittest import mock

import numpy as np

from orthogonal_dfa.l_star import suffix_groups


class TestNearestToAnchorGroup(unittest.TestCase):
    def test_the_anchor_group_comes_before_rows_nearer_its_mean(self):
        # Two groups that differ only on 20 of 600 columns: read noise puts many
        # of the other group's rows nearer the anchor group's mean than its own.
        rng = np.random.default_rng(0)
        columns, few = 600, 20
        rates = np.full(columns, 0.3)
        other = rates.copy()
        other[:few] = 0.7
        rows = np.vstack(
            [rng.random((300, columns)) < rates, rng.random((300, columns)) < other]
        ).astype(float)
        groups = [np.arange(300), np.arange(300, 600)]
        anchor = (rng.random(columns) < rates).astype(float)
        everything = np.ones(columns, dtype=bool)
        with mock.patch.object(suffix_groups, "coherent_groups", return_value=groups):
            order = suffix_groups.nearest_to_anchor_group(
                rows,
                anchor,
                [everything],
                everything,
                (0.3, 0.7),
                k=2,
                alpha=0.01,
                rng=rng,
            )
        self.assertEqual(set(order[:300]), set(groups[0]))


if __name__ == "__main__":
    unittest.main()
