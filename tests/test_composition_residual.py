import unittest

import numpy as np
import torch

from orthogonal_dfa.data.exon import RawExon
from orthogonal_dfa.l_star.examples.composition_residual import (
    bow_features,
    fit_composition_residual,
)
from orthogonal_dfa.l_star.examples.spliceai_oracle import (
    SpliceModelOracle,
    flanks,
    median_threshold,
    run_over_middles,
)
from orthogonal_dfa.spliceai.exon_score import FLANK_MARGIN, SpliceAIExonScore
from orthogonal_dfa.spliceai.module import SpliceAIModule

SMALL_CL = 80
FULL_LENGTH = 30


def small_score_model_and_exon(cl=SMALL_CL):
    torch.manual_seed(0)
    score_model = SpliceAIExonScore(SpliceAIModule(window=cl)).eval()
    exon = RawExon(
        cl,
        np.random.default_rng(0)
        .integers(0, 4, size=cl + 2 * FLANK_MARGIN + FULL_LENGTH)
        .tolist(),
    )
    return score_model, exon


class TestBowFeatures(unittest.TestCase):
    def test_shape_and_frequencies(self):
        features = bow_features([[0, 1, 2, 3], [3, 3, 3]], n_max=2)
        self.assertEqual(features.shape, (2, 4 + 16))
        np.testing.assert_allclose(features[0, :4], 0.25)  # ACGT -> each base 1/4
        np.testing.assert_allclose(features[1, :4], [0, 0, 0, 1])  # TTT -> T only


class TestCompositionResidualScore(unittest.TestCase):
    def setUp(self):
        self.score_model, self.exon = small_score_model_and_exon()
        self.flank_l, self.flank_r = flanks(self.exon)

    def _fit(self, **kw):
        kw.setdefault("n_max", 2)
        return fit_composition_residual(
            self.score_model, self.exon, per_bin=300, chunk=64, device="cpu", **kw
        )

    def _scores(self, model, middles):
        return run_over_middles(
            model, self.flank_l, self.flank_r, middles, device="cpu", chunk=64
        )

    def test_single_bin_at_one_length(self):
        residual = self._fit(len_lo=20, len_hi=21, bin_width=1)
        self.assertEqual(len(residual._bins), 1)  # pylint: disable=protected-access

    def test_forward_subtracts_a_linear_function_of_composition(self):
        residual = self._fit(len_lo=20, len_hi=21, bin_width=1)
        middles = np.random.default_rng(1).integers(0, 4, size=(256, 20)).tolist()
        subtracted = self._scores(self.score_model, middles) - self._scores(
            residual, middles
        )
        # what was subtracted is exactly s_mean + (bow - f_mean) @ beta, i.e. an
        # affine function of the bag-of-k-mers features, so it fits with no residual.
        design = np.column_stack([np.ones(len(middles)), bow_features(middles, 2)])
        coef, *_ = np.linalg.lstsq(design, subtracted, rcond=None)
        np.testing.assert_allclose(design @ coef, subtracted, atol=1e-4)

    def test_oracle_balanced_across_lengths(self):
        residual = self._fit(len_lo=20, len_hi=40, bin_width=5)
        threshold = median_threshold.function(
            residual, self.exon, 30, count=256, chunk=64, device="cpu"
        )
        oracle = SpliceModelOracle(
            self.exon, residual, threshold, device="cpu", chunk=64
        )
        for length in (25, 35):
            fresh = (
                np.random.default_rng(length)
                .integers(0, 4, size=(256, length))
                .tolist()
            )
            self.assertAlmostEqual(
                oracle.membership_queries(fresh).mean(), 0.5, delta=0.2
            )


if __name__ == "__main__":
    unittest.main()
