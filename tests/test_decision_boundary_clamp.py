"""A cluster that lands all on one side estimates a boundary off the scale the
rates live on: ``identify_cluster_around`` reads the boundary off the accepting
group alone when it finds no rejects, and off the rejecting group alone when it
finds no accepts, stepping a signal off that group's mean.
"""

import unittest

import numpy as np

from orthogonal_dfa.l_star.cluster import identify_cluster_around
from tests.lstar_common import cluster_pst

SIGNAL = 0.3
NUM_SUFFIXES, NUM_PREFIXES = 8, 16


def _boundary(masks, signal=SIGNAL):
    _, boundary = identify_cluster_around(cluster_pst(masks, signal), 0, 4, 0.5)
    return boundary


class TestDecisionBoundaryClamp(unittest.TestCase):
    def test_all_accepts_stays_a_probability(self):
        boundary = _boundary(np.ones((NUM_SUFFIXES, NUM_PREFIXES), dtype=np.int8))
        self.assertLessEqual(boundary + SIGNAL, 1)
        self.assertGreaterEqual(boundary - SIGNAL, 0)

    def test_all_rejects_stays_a_probability(self):
        boundary = _boundary(np.zeros((NUM_SUFFIXES, NUM_PREFIXES), dtype=np.int8))
        self.assertLessEqual(boundary + SIGNAL, 1)
        self.assertGreaterEqual(boundary - SIGNAL, 0)

    def test_a_separated_cluster_is_left_alone(self):
        # Half the prefixes answer one way and half the other, so the estimate
        # already sits between the two groups and the clamp has nothing to do.
        masks = np.zeros((NUM_SUFFIXES, NUM_PREFIXES), dtype=np.int8)
        masks[:, : NUM_PREFIXES // 2] = 1
        self.assertAlmostEqual(_boundary(masks), 0.5)

    def test_a_one_sided_cluster_clears_the_class_it_found(self):
        # Three of the four clustered rows accept, so the single group's mean is
        # 0.75 and the boundary has to sit a signal below it rather than on it.
        masks = np.ones((NUM_SUFFIXES, NUM_PREFIXES), dtype=np.int8)
        masks[3:] = 0
        self.assertAlmostEqual(_boundary(masks, 0.2), 0.55)

    def test_a_weak_signal_clamps_further_out(self):
        wide = _boundary(np.ones((NUM_SUFFIXES, NUM_PREFIXES), dtype=np.int8), 0.3)
        narrow = _boundary(np.ones((NUM_SUFFIXES, NUM_PREFIXES), dtype=np.int8), 0.1)
        self.assertGreater(narrow, wide)


if __name__ == "__main__":
    unittest.main()
