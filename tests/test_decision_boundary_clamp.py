"""``identify_cluster_around`` reads the boundary off the separation between the
two groups, so a cluster that lands all on one side has nothing to read and keeps
the boundary it came in with.  A lopsided two-sided estimate is still clamped to
the scale the rates live on.
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


def _boundary(masks, signal=SIGNAL, incoming=0.5):
    pst = SimpleNamespace(
        table=_Table(masks), config=SimpleNamespace(min_signal_strength=signal)
    )
    _, boundary = identify_cluster_around(pst, 0, 4, incoming)
    return boundary


def _lopsided():
    """Clustered prefix means of 1.0 on twelve columns and 0.5 on four, which the
    incoming boundary of 0.5 splits twelve against four for a midpoint of 0.75."""
    masks = np.ones((NUM_SUFFIXES, NUM_PREFIXES), dtype=np.int8)
    masks[2:4, 12:] = 0
    masks[4:] = 0
    return masks


class TestDecisionBoundaryClamp(unittest.TestCase):
    def test_all_accepts_keeps_the_incoming_boundary(self):
        masks = np.ones((NUM_SUFFIXES, NUM_PREFIXES), dtype=np.int8)
        self.assertAlmostEqual(_boundary(masks, incoming=0.4), 0.4)

    def test_all_rejects_keeps_the_incoming_boundary(self):
        masks = np.zeros((NUM_SUFFIXES, NUM_PREFIXES), dtype=np.int8)
        self.assertAlmostEqual(_boundary(masks, incoming=0.6), 0.6)

    def test_a_separated_cluster_is_left_alone(self):
        # Half the prefixes answer one way and half the other, so the estimate
        # already sits between the two groups and the clamp has nothing to do.
        masks = np.zeros((NUM_SUFFIXES, NUM_PREFIXES), dtype=np.int8)
        masks[:, : NUM_PREFIXES // 2] = 1
        self.assertAlmostEqual(_boundary(masks), 0.5)

    def test_a_lopsided_estimate_stays_a_probability(self):
        boundary = _boundary(_lopsided())
        self.assertLessEqual(boundary + SIGNAL, 1)
        self.assertGreaterEqual(boundary - SIGNAL, 0)

    def test_a_weak_signal_clamps_further_out(self):
        wide = _boundary(_lopsided(), 0.3)
        narrow = _boundary(_lopsided(), 0.1)
        self.assertGreater(narrow, wide)


if __name__ == "__main__":
    unittest.main()
