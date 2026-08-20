"""The marginal-benefit ranking is validated against a mock score with a *known*
context-defined motif -- no SpliceAI, no GPU."""

import unittest

import numpy as np
from permacache import no_cache_global

from orthogonal_dfa.analysis.nonlinear_motif_miner import (
    ScoreOracle,
    marginal_records_until,
)

L = 24


class PlantedMotifOracle(ScoreOracle):
    """Per-position weights plus a bonus whenever `motif` appears anywhere."""

    def __init__(self, motif, alphabet="ACGT", bonus=3.0):
        self._alphabet = alphabet
        self.motif = motif
        self.bonus = bonus
        self.weights = np.random.default_rng(0).standard_normal((L, len(alphabet)))

    @property
    def alphabet(self):
        return self._alphabet

    @property
    def length(self):
        return L

    def scores(self, seqs):
        out = []
        for s in seqs:
            v = float(sum(self.weights[i, s[i]] for i in range(len(s))))
            width = len(self.motif)
            if any(
                tuple(s[j : j + width]) == self.motif for j in range(len(s) - width + 1)
            ):
                v += self.bonus
            out.append(v)
        return np.array(out)

    def __permacache_hash__(self):
        return [self._alphabet, self.motif, self.bonus]


class MinerTestCase(unittest.TestCase):
    """Scans here must recompute; a cached round would let a broken change pass."""

    def setUp(self):
        self.enterContext(no_cache_global())

    def rank(self, oracle, max_k, n_contexts):
        # 1e-9 is unreachable, so the scan runs to exactly n_contexts.
        records, info = marginal_records_until(
            oracle,
            max_k,
            1e-9,
            contexts_per_round=n_contexts,
            max_contexts=n_contexts,
        )
        return {r.motif: i for i, r in enumerate(records)}, records, info


class TestMarginalRanking(MinerTestCase):
    def test_a_longer_motif_that_only_extends_a_shorter_one_sinks(self):
        # CG carries the whole effect, so 3-mers that merely contain it add no marginal
        # benefit and must rank BELOW CG.
        order, _, _ = self.rank(PlantedMotifOracle((1, 2)), 3, 1500)
        self.assertLess(order["CG"], order["ACG"])
        self.assertLess(order["CG"], order["CGT"])
        self.assertLess(order["CG"], 3)


class TestNonDNAAlphabet(MinerTestCase):
    def test_binary_alphabet(self):
        order, records, _ = self.rank(
            PlantedMotifOracle((0, 1), alphabet="ab"), 3, 1500
        )
        self.assertEqual(len(records), 2 + 4 + 8)
        self.assertLess(order["ab"], order["aab"])
        self.assertLess(order["ab"], order["abb"])


class TestStopping(MinerTestCase):
    def test_stops_at_max_contexts(self):
        _, _, info = self.rank(PlantedMotifOracle((1, 2)), 3, 800)
        self.assertEqual(info["n_contexts"], 800)
        self.assertGreater(info["bound"], 1e-9)  # cap hit, target not met

    def test_a_reachable_target_stops_early(self):
        _, info = marginal_records_until(
            PlantedMotifOracle((1, 2)),
            3,
            10.0,
            contexts_per_round=200,
            max_contexts=4000,
        )
        self.assertEqual(info["n_contexts"], 200)
        self.assertLessEqual(info["bound"], 10.0)


class TestOracleIsHashable(MinerTestCase):
    def test_permacache_hash_distinguishes_oracles(self):
        from permacache import stable_hash

        self.assertNotEqual(
            stable_hash(PlantedMotifOracle((1, 2))),
            stable_hash(PlantedMotifOracle((0, 1))),
        )


if __name__ == "__main__":
    unittest.main()
