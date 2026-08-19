"""The marginal-benefit ranking is validated against a mock score with a *known*
context-defined motif -- no SpliceAI, no GPU."""
import unittest

import numpy as np

from orthogonal_dfa.analysis.nonlinear_motif_miner import (
    marginal_motif_records,
    marginal_records_until,
    replicates_for_max_error,
    sample_contexts,
)

L = 24


class TestMarginalMotifRecords(unittest.TestCase):
    def test_a_longer_motif_that_only_extends_a_shorter_one_sinks(self):
        # Score adds a bonus whenever "CG" (1,2) appears ANYWHERE, plus per-position noise.
        # CG carries the whole effect, so 3-mers that merely contain it (ACG, CGT) add no
        # marginal benefit and must rank BELOW CG.
        target = (1, 2)  # CG
        W = np.random.default_rng(0).standard_normal((L, 4))

        def score(middles):
            out = []
            for m in middles:
                v = float(sum(W[i, m[i]] for i in range(len(m))))
                if any(tuple(m[j : j + 2]) == target for j in range(len(m) - 1)):
                    v += 3.0
                out.append(v)
            return np.array(out)

        contexts = sample_contexts(L, 3, 1500, seed=1)
        records = marginal_motif_records(score, contexts, max_k=3)
        rank = {r.motif: i for i, r in enumerate(records)}  # 0 = highest marginal
        self.assertLess(rank["CG"], rank["ACG"])
        self.assertLess(rank["CG"], rank["CGT"])
        # CG itself should be at (or very near) the top by marginal benefit
        self.assertLess(rank["CG"], 3)


class TestReplicatesForMaxError(unittest.TestCase):
    def test_matches_the_closed_form(self):
        import math
        std, M, eps, delta = 0.5, 4096, 0.02, 0.05
        z = math.sqrt(2 * math.log(M)) + math.sqrt(2 * math.log(1 / delta))
        expected = math.ceil((std * z / eps) ** 2)
        self.assertEqual(replicates_for_max_error(std, M, eps, delta=delta), expected)

    def test_monotonicities(self):
        base = replicates_for_max_error(0.5, 4096, 0.02)
        self.assertGreater(replicates_for_max_error(0.5, 4096, 0.01), base)  # tighter target
        self.assertGreater(replicates_for_max_error(1.0, 4096, 0.02), base)  # bigger std
        self.assertGreater(replicates_for_max_error(0.5, 4 ** 8, 0.02), base)  # more motifs
        self.assertLess(replicates_for_max_error(0.5, 64, 0.02), base)  # fewer motifs


class TestMarginalRecordsUntil(unittest.TestCase):
    def _cg_score(self):
        W = np.random.default_rng(0).standard_normal((L, 4))

        def score(middles):
            out = []
            for m in middles:
                v = float(sum(W[i, m[i]] for i in range(len(m))))
                if any(tuple(m[j : j + 2]) == (1, 2) for j in range(len(m) - 1)):
                    v += 3.0
                out.append(v)
            return np.array(out)

        return score

    def test_stops_at_max_contexts_and_still_ranks_cg_top(self):
        score = self._cg_score()
        make = lambda seed, n: sample_contexts(L, 3, n, seed=seed)
        records, info = marginal_records_until(
            score, make, max_k=3, target_error=1e-6, batch=400, max_contexts=800
        )
        self.assertEqual(info["n_contexts"], 800)  # unreachable target -> hits the cap
        rank = {r.motif: i for i, r in enumerate(records)}
        self.assertLess(rank["CG"], rank["ACG"])

    def test_looser_target_uses_no_more_contexts(self):
        score = self._cg_score()
        make = lambda seed, n: sample_contexts(L, 3, n, seed=seed)
        _, loose = marginal_records_until(
            score, make, max_k=3, target_error=1.0, batch=400, max_contexts=4000
        )
        _, tight = marginal_records_until(
            score, make, max_k=3, target_error=1e-6, batch=400, max_contexts=4000
        )
        self.assertLessEqual(loose["n_contexts"], tight["n_contexts"])


if __name__ == "__main__":
    unittest.main()
