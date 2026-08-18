"""The perturbation math is validated against a mock score with a *known* additive
part and a *known* nonlinear motif -- no SpliceAI, no GPU."""
import unittest

import numpy as np

from orthogonal_dfa.analysis.nonlinear_motif_miner import (
    additive_prediction,
    context_motif_significance,
    harvest,
    motif_effects,
    sample_backgrounds,
    sample_contexts,
    single_base_ism,
)

L = 24
P0 = 7
MOTIF = (0, 1, 2)  # "ACG"
BONUS = 5.0


def make_mock_score(seed=0):
    """score = per-position-base additive weights + a BONUS when MOTIF sits at P0."""
    W = np.random.default_rng(seed).standard_normal((L, 4))

    def score(middles):
        out = []
        for m in middles:
            v = float(sum(W[i, m[i]] for i in range(len(m))))
            if tuple(m[P0 : P0 + len(MOTIF)]) == MOTIF:
                v += BONUS
            out.append(v)
        return np.array(out)

    return score, W


class TestSingleBaseISM(unittest.TestCase):
    def test_ism_recovers_additive_weights_up_to_row_mean(self):
        score, W = make_mock_score()
        bg = sample_backgrounds(L, 4000, seed=1)
        ism = single_base_ism(score, bg)
        # ism[i,b] == W[i,b] - mean_b W[i,b]  (population-mean over backgrounds)
        expected = W - W.mean(1, keepdims=True)
        # the P0 motif region carries a small extra ISM signal; check the clean columns
        clean = [i for i in range(L) if i < P0 - 2 or i > P0 + 2]
        np.testing.assert_allclose(ism[clean], expected[clean], atol=0.1)


class TestMotifEpistasis(unittest.TestCase):
    def test_motif_at_p0_has_nonlinear_effect_near_bonus(self):
        score, _ = make_mock_score()
        bg = sample_backgrounds(L, 4000, seed=2)
        ism = single_base_ism(score, bg)
        eff = motif_effects(score, bg, [list(MOTIF)], [P0])[0, 0]
        add = additive_prediction(ism, list(MOTIF), P0)
        nonlinear = eff - add
        # The epistatic component recovers the bulk of the BONUS.  It comes in a little
        # under the raw BONUS because ISM partially attributes the motif's bonus to its
        # single bases (setting P0->A completes the motif whenever P0+1,P0+2 are already
        # C,G), which inflates the additive prediction -- a real property of the method.
        self.assertGreater(nonlinear, 0.7 * BONUS)
        self.assertLess(nonlinear, 1.05 * BONUS)

    def test_a_non_bonus_motif_is_additive(self):
        score, _ = make_mock_score()
        bg = sample_backgrounds(L, 4000, seed=3)
        ism = single_base_ism(score, bg)
        other = (3, 3, 3)  # "TTT" at a position away from P0 -> no epistasis
        pos = 2
        eff = motif_effects(score, bg, [list(other)], [pos])[0, 0]
        nonlinear = eff - additive_prediction(ism, list(other), pos)
        self.assertAlmostEqual(nonlinear, 0.0, delta=0.3)


class TestHarvest(unittest.TestCase):
    def test_harvest_ranks_the_true_motif_top_by_nonlinearity(self):
        score, _ = make_mock_score()
        _, hits = harvest(score, L, n_backgrounds=3000, motif_k=3,
                          top_positions=L, seed=4)
        top = max(hits, key=lambda h: abs(h.nonlinear))
        self.assertEqual(top.motif, "ACG")
        self.assertEqual(top.position, P0)
        self.assertGreater(top.nonlinear, 3.0)


class TestContextMotifSignificance(unittest.TestCase):
    def test_a_context_independent_motif_is_significant_anywhere(self):
        # A score that adds a bonus whenever "GTA" appears ANYWHERE (position-agnostic,
        # context-defined) plus per-position additive noise.  The in-context test should
        # flag GTA as significantly raising the score vs alternatives, at any position.
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
        top = max(stats, key=lambda s: s.tstat)
        self.assertEqual(top.motif, "GTA")
        self.assertGreater(top.tstat, 5.0)  # clearly significant
        self.assertGreater(top.effect, 0)


if __name__ == "__main__":
    unittest.main()
