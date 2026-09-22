"""Suffixes tied with the seed must not push it out of its own cluster.

The loss ranking suffixes is a count of disagreements with the cluster center,
so the less noise an oracle carries the more suffixes tie at the seed's own
loss.  Ranking within that tie by anything but the seed drops it from a cluster
it heads, leaving a family of one that no amount of further sampling grows.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from orthogonal_dfa.l_star.cluster import identify_cluster_around

SIGNAL = 0.3
NUM_PREFIXES = 16
#: More suffixes tied with the seed than the cluster holds, so which of them it
#: keeps is decided inside the tie rather than by the loss.
NUM_SUFFIXES, COUNT = 200, 150


class _Table:
    def __init__(self, masks):
        self._masks = masks
        self.representative = np.ones(masks.shape[1], dtype=bool)

    def fully_observed(self):
        return np.arange(self._masks.shape[0])

    def observed_masks(self, rows, prefixes):
        return self._masks[np.asarray(rows)][:, prefixes]


def _cluster(masks, seed):
    pst = SimpleNamespace(
        table=_Table(masks), config=SimpleNamespace(min_signal_strength=SIGNAL)
    )
    vs, _ = identify_cluster_around(pst, seed, COUNT, 0.5)
    return vs


def _agreeing(num_suffixes=NUM_SUFFIXES):
    masks = np.zeros((num_suffixes, NUM_PREFIXES), dtype=np.int8)
    masks[:, : NUM_PREFIXES // 2] = 1
    return masks


class TestClusterSeedTies(unittest.TestCase):
    def test_a_seed_ranked_below_the_tie_keeps_its_cluster(self):
        vs = _cluster(_agreeing(), NUM_SUFFIXES - 1)
        self.assertIn(NUM_SUFFIXES - 1, vs)
        self.assertEqual(len(vs), COUNT)

    def test_every_seed_in_the_tie_keeps_its_cluster(self):
        for seed in range(0, NUM_SUFFIXES, 17):
            with self.subTest(seed=seed):
                vs = _cluster(_agreeing(), seed)
                self.assertIn(seed, vs)
                self.assertEqual(len(vs), COUNT)

    def test_a_center_drifting_off_the_seed_still_stops(self):
        # Ranking the seed first must not let the center walk away from it.
        # Here every other suffix disagrees with the seed on every prefix, so
        # the cluster the loop would converge on excludes it; what comes back
        # is the last one it headed.
        masks = _agreeing()
        masks[0] = 1 - masks[0]
        self.assertIn(0, _cluster(masks, 0))
