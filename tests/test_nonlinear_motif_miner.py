"""The in-context significance math is validated against a mock score with a *known*
context-defined motif -- no SpliceAI, no GPU."""
import unittest

import numpy as np

from orthogonal_dfa.analysis.nonlinear_motif_miner import (
    context_motif_significance,
    sample_contexts,
)

L = 24


class TestContextMotifSignificance(unittest.TestCase):
    def test_a_context_defined_motif_has_the_largest_magnitude_anywhere(self):
        # A score that adds a bonus whenever "GTA" appears ANYWHERE (position-agnostic,
        # context-defined) plus per-position additive noise.  The in-context test should
        # flag GTA as having the largest effect magnitude vs alternatives, at any position.
        target = (2, 3, 0)  # GTA
        W = np.random.default_rng(0).standard_normal((L, 4))

        def score(middles):
            out = []
            for m in middles:
                v = float(sum(W[i, m[i]] for i in range(len(m))))
                if any(tuple(m[j : j + 3]) == target for j in range(len(m) - 2)):
                    v += 3.0
                out.append(v)
            return np.array(out)

        contexts = sample_contexts(L, 3, 2000, seed=1)
        stats = context_motif_significance(score, contexts, 3)
        top = max(stats, key=lambda s: s.magnitude)
        self.assertEqual(top.motif, "GTA")
        self.assertGreater(top.mag_z, 3.0)  # clearly above the average k-mer


if __name__ == "__main__":
    unittest.main()
