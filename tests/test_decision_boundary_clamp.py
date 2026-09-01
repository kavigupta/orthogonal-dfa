"""A cluster that lands all on one side estimates a boundary off the scale the
rates live on: ``identify_cluster_around`` reads the boundary off the accepting
group alone when it finds no rejects, and off the rejecting group alone when it
finds no accepts.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from orthogonal_dfa.l_star.cluster import identify_cluster_around

SIGNAL = 0.3
NUM_SUFFIXES, NUM_PREFIXES = 8, 16


class _Table:
    def __init__(self, masks):
        self._masks = masks
        self.representative = np.ones(masks.shape[1], dtype=bool)

    def fully_observed(self):
        return np.arange(self._masks.shape[0])

    def observed_masks(self, rows, prefixes):
        return self._masks[np.asarray(rows)][:, prefixes]

    def strata_masks(self):
        return {"baseline": self.representative}


def _boundary(masks, signal=SIGNAL):
    pst = SimpleNamespace(
        table=_Table(masks), config=SimpleNamespace(min_signal_strength=signal)
    )
    _, boundary = identify_cluster_around(pst, 0, 4, 0.5)
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

    def test_a_weak_signal_clamps_further_out(self):
        wide = _boundary(np.ones((NUM_SUFFIXES, NUM_PREFIXES), dtype=np.int8), 0.3)
        narrow = _boundary(np.ones((NUM_SUFFIXES, NUM_PREFIXES), dtype=np.int8), 0.1)
        self.assertGreater(narrow, wide)


if __name__ == "__main__":
    unittest.main()
