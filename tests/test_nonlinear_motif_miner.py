"""The marginal-benefit ranking is validated against a mock score with a *known*
context-defined motif -- no SpliceAI, no GPU."""
import unittest

import numpy as np

from orthogonal_dfa.analysis.nonlinear_motif_miner import (
    marginal_motif_records,
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


if __name__ == "__main__":
    unittest.main()
