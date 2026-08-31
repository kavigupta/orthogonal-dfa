import unittest

import numpy as np

from orthogonal_dfa.l_star.counterexample_synthesis import classify_pool
from orthogonal_dfa.l_star.midfix_tree import MidfixTree

ACCEPT, REJECT = 0.7, 0.3

#: Family membership of each node's distinguishers, per prefix.
DECISIONS = {
    b"": np.array([0.9, 0.1, 0.5, 0.9, 0.1, 0.1, 0.9, 0.5]),
    b"m": np.array([0.4, 0.9, 0.4, 0.4, 0.2, 0.5, 0.4, 0.4]),
}


class _StubTracker:
    """Answers from DECISIONS, recording the prefixes each read asked for."""

    def __init__(self):
        self.asked = []

    def compute_decision_from_strings(self, vs, subset_prefixes):
        midfix = vs[0][: -len(b"a")]
        self.asked.append(np.asarray(subset_prefixes).copy())
        return DECISIONS[midfix][subset_prefixes]


def _tree():
    tree = MidfixTree([b"a", b"b"])
    tree.split(1, b"m")
    return tree


def _classify(mask):
    pst = _StubTracker()
    leaves = classify_pool(pst, _tree(), accept=ACCEPT, reject=REJECT, prefixes=mask)
    return leaves, pst.asked


class TestClassifyPoolScope(unittest.TestCase):
    def test_masking_gives_the_same_leaves_as_classifying_everything(self):
        mask = np.array([1, 0, 1, 1, 0, 1, 0, 1], dtype=bool)
        every = np.ones(len(mask), dtype=bool)
        self.assertEqual(list(_classify(mask)[0]), list(_classify(every)[0][mask]))

    def test_reads_only_the_prefixes_asked_for(self):
        mask = np.array([1, 0, 1, 1, 0, 1, 0, 1], dtype=bool)
        _, asked = _classify(mask)
        self.assertTrue(asked)
        for subset in asked:
            self.assertEqual(list(subset), list(mask))

    def test_a_prefix_no_node_places_stays_undecided(self):
        every = np.ones(8, dtype=bool)
        # prefix 2 sits in the root's band, prefix 5 in the child's.
        self.assertEqual(list(_classify(every)[0]), [0, 1, -1, 0, 2, -1, 0, -1])
