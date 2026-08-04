import unittest
import warnings

import numpy as np
import torch

from orthogonal_dfa.data.exon import RawExon
from orthogonal_dfa.l_star.examples.composition_residual import (
    _fit_bin,
    _fit_composition_bins,
    _residual_module,
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
# RawExon.random_text_length = len(text) - (cl + 4), so this is the query length.
QUERY_LENGTH = 30


def small_score_model_and_exon(cl=SMALL_CL):
    torch.manual_seed(0)
    score_model = SpliceAIExonScore(SpliceAIModule(window=cl)).eval()
    exon = RawExon(
        cl,
        np.random.default_rng(0)
        .integers(0, 4, size=cl + 2 * FLANK_MARGIN + QUERY_LENGTH)
        .tolist(),
    )
    return score_model, exon


class TestBowFeatures(unittest.TestCase):
    def test_shape_and_frequencies(self):
        features = bow_features([[0, 1, 2, 3], [3, 3, 3]], n_max=2)
        self.assertEqual(features.shape, (2, 4 + 16))
        np.testing.assert_allclose(features[0, :4], 0.25)  # ACGT -> each base 1/4
        np.testing.assert_allclose(features[1, :4], [0, 0, 0, 1])  # TTT -> T only


class TestFitBin(unittest.TestCase):
    def test_ridge_is_invariant_to_feature_scale(self):
        # The penalty must not depend on a feature's raw scale: otherwise rarer
        # (lower-variance) higher-order k-mers get shrunk harder than 1-mers.  The
        # standardized ridge makes the fitted prediction invariant to rescaling any
        # column -- so its coefficient rescales inversely and nothing else moves.
        rng = np.random.default_rng(0)
        feats = rng.random((500, 6))
        scores = feats @ rng.random(6) + 0.01 * rng.standard_normal(500)
        _, _, beta = _fit_bin(feats, scores, ridge=1.0)
        scaled = feats.copy()
        scaled[:, 2] *= 7.0
        _, _, beta_scaled = _fit_bin(scaled, scores, ridge=1.0)
        np.testing.assert_allclose(beta_scaled[2], beta[2] / 7.0, rtol=1e-6)
        np.testing.assert_allclose(
            (feats - feats.mean(0)) @ beta,
            (scaled - scaled.mean(0)) @ beta_scaled,
            rtol=1e-6,
        )

    def test_effective_penalty_shrinks_with_more_samples(self):
        # lambda ~ D/n, so a fixed correlation strength regularizes less with more
        # data: doubling the sample count moves beta strictly toward OLS (ridge=0).
        rng = np.random.default_rng(1)
        feats = rng.random((400, 6))
        scores = feats @ rng.random(6) + 0.1 * rng.standard_normal(400)
        _, _, ols = _fit_bin(feats, scores, ridge=0.0)
        _, _, small = _fit_bin(feats, scores, ridge=1.0)
        _, _, large = _fit_bin(np.tile(feats, (2, 1)), np.tile(scores, 2), ridge=1.0)
        self.assertLess(np.linalg.norm(large - ols), np.linalg.norm(small - ols))


class TestCompositionResidualScore(unittest.TestCase):
    def setUp(self):
        self.score_model, self.exon = small_score_model_and_exon()
        self.flank_l, self.flank_r = flanks(self.exon)

    def _fit(self, *, n_max=2, **band):
        # cache=False recomputes: the tests must see the fit as the math currently is,
        # not a stale permacache entry keyed only by (model, exon, params).
        return fit_composition_residual(
            self.score_model,
            self.exon,
            n_max=n_max,
            per_bin=300,
            device="cpu",
            chunk=64,
            cache=False,
            **band,
        )

    def _scores(self, model, middles):
        return run_over_middles(
            model, self.flank_l, self.flank_r, middles, device="cpu", chunk=64
        )

    def test_single_bin_at_one_length(self):
        # pylint: disable=protected-access
        # the documented single-length form: len_hi = len_lo + 1 (len_hi exclusive).
        residual = self._fit(len_lo=30, len_hi=31, bin_width=1)
        self.assertEqual(residual._f_means.shape[0], 1)
        scored = self._scores(residual, [[0, 1, 2] * 10])
        self.assertEqual(scored.shape, (1,))
        self.assertFalse(np.isnan(residual.composition_r2))

    def test_empty_length_range_is_rejected(self):
        # len_hi == len_lo yields zero bins; the fitter must fail loudly, not return a
        # module with no bins (nan r2, IndexError later).
        with self.assertRaises(AssertionError):
            _fit_composition_bins.function(
                self.score_model,
                self.exon,
                len_lo=30,
                len_hi=30,
                bin_width=1,
                per_bin=8,
            )

    def test_fit_time_rejects_a_band_missing_the_query_length(self):
        # the public entry point asserts the exon's query length is in the band.
        with self.assertRaises(AssertionError):
            fit_composition_residual(
                self.score_model, self.exon, len_lo=5, len_hi=10, per_bin=300
            )

    def test_out_of_band_query_length_warns(self):
        residual = self._fit(len_lo=30, len_hi=35, bin_width=5)
        with self.assertWarns(UserWarning):
            self._scores(residual, [[0] * 60])

    def test_forward_subtracts_a_linear_function_of_composition(self):
        residual = self._fit(len_lo=30, len_hi=31, bin_width=1)
        middles = np.random.default_rng(1).integers(0, 4, size=(256, 30)).tolist()
        subtracted = self._scores(self.score_model, middles) - self._scores(
            residual, middles
        )
        # what was subtracted is s_mean + (bow - f_mean) @ beta, i.e. an affine
        # function of the bag-of-k-mers features, so it fits with no residual.
        design = np.column_stack([np.ones(len(middles)), bow_features(middles, 2)])
        coef, *_ = np.linalg.lstsq(design, subtracted, rcond=None)
        np.testing.assert_allclose(design @ coef, subtracted, atol=1e-4)

    def test_fit_survives_state_dict_round_trip(self):
        residual = self._fit(len_lo=30, len_hi=31, bin_width=1)
        middles = np.random.default_rng(2).integers(0, 4, size=(8, 30)).tolist()
        before = self._scores(residual, middles)
        zeroed = _residual_module(
            self.score_model,
            self.exon,
            2,
            dict(
                edge0=0.0,
                step=1.0,
                # pylint: disable=protected-access
                f_means=np.zeros_like(residual._f_means.numpy()),
                s_means=np.zeros_like(residual._s_means.numpy()),
                betas=np.zeros_like(residual._betas.numpy()),
                r2=0.0,
            ),
            "cpu",
        )
        with warnings.catch_warnings():  # the zeroed band [0, 1) is out of band by design
            warnings.simplefilter("ignore")
            self.assertFalse(np.allclose(self._scores(zeroed, middles), before))
        zeroed.load_state_dict(residual.state_dict())
        np.testing.assert_allclose(self._scores(zeroed, middles), before, rtol=1e-6)

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
