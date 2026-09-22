import unittest
import warnings

import numpy as np
from permacache import no_cache_global

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
from tests.spliceai_small import random_middles, small_score_model_and_exon


def every_kmer_frequency(strings, k_max):
    """All 4**k frequencies per k, including the ones bow_features omits."""
    rows = []
    for s in strings:
        s, row = np.frombuffer(s, dtype=np.uint8).astype(np.int64), []
        for k in range(1, k_max + 1):
            ids = np.zeros(len(s) - k + 1, dtype=np.int64)
            for j in range(k):
                ids = ids * 4 + s[j : len(s) - k + 1 + j]
            row.append(np.bincount(ids, minlength=4**k) / (len(s) - k + 1))
        rows.append(np.concatenate(row))
    return np.array(rows)


class TestBowFeatures(unittest.TestCase):
    def test_shape_and_frequencies(self):
        features = bow_features([bytes([0, 1, 2, 3]), bytes([3, 3, 3])], k_max=2)
        self.assertEqual(features.shape, (2, 3 + 15))  # last k-mer of each k omitted
        np.testing.assert_allclose(features[0, :3], 0.25)  # ACGT -> A, C, G each 1/4
        np.testing.assert_allclose(features[1, :3], 0)  # TTT is all of the omitted T

    def test_omitted_kmer_is_one_minus_the_rest(self):
        mids, _ = random_middles(16, 40, seed=3)
        features = bow_features(mids, 2)
        full = every_kmer_frequency(mids, 2)
        np.testing.assert_allclose(1 - features[:, :3].sum(1), full[:, 3], atol=1e-6)
        np.testing.assert_allclose(1 - features[:, 3:].sum(1), full[:, 19], atol=1e-6)


class TestFitBin(unittest.TestCase):
    def test_omitting_a_kmer_per_block_loses_nothing(self):
        # A block's frequencies sum to 1, so the omitted k-mer is one minus the rest:
        # with an intercept the 18-wide design still spans all 20 frequencies.  A
        # target exactly affine in all 20 is therefore fit with no residual.
        mids, rng = random_middles(500, 40, seed=1)
        scores = every_kmer_frequency(mids, 2) @ rng.standard_normal(20) + 3.0
        feats = bow_features(mids, 2).astype(np.float64)
        intercept, beta = _fit_bin(feats, scores)
        np.testing.assert_allclose(intercept + feats @ beta, scores, rtol=1e-6)

    def test_rejects_per_bin_too_small_to_fit_unregularized(self):
        # k_max=2 is 18 features wide -> needs 360 middles; 100 must not fit.
        with self.assertRaises(AssertionError):
            _fit_composition_bins.function(
                *small_score_model_and_exon(),
                n_max=2,
                len_lo=30,
                len_hi=31,
                bin_width=1,
                per_bin=100,
                device="cpu",
                chunk=64,
            )


class TestCompositionResidualScore(unittest.TestCase):
    def setUp(self):
        self.score_model, self.exon = small_score_model_and_exon()
        self.flank_l, self.flank_r = flanks(self.exon)

    def _fit(self, *, n_max=2, **band):
        # The tests must see the fit as the math currently is, not a stale permacache
        # entry keyed only by (model, exon, params).
        with no_cache_global():
            return fit_composition_residual(
                self.score_model,
                self.exon,
                n_max=n_max,
                per_bin=400,  # >= MIN_SAMPLES_PER_PARAMETER * free_parameters(2) == 360
                device="cpu",
                chunk=64,
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
        self.assertEqual(residual._betas.shape[0], 1)
        scored = self._scores(residual, [bytes([0, 1, 2]) * 10])
        self.assertEqual(scored.shape, (1,))
        self.assertEqual(residual.composition_r2s.shape, (1,))
        self.assertFalse(np.isnan(residual.composition_r2))

    def test_empty_length_range_is_rejected(self):
        # len_hi == len_lo yields zero bins; the fitter must fail loudly, not return a
        # module with no bins (nan r2, IndexError later).
        # per_bin/n_max are kept valid so only the empty-range assert can fire.
        with self.assertRaisesRegex(AssertionError, "at least one length bin"):
            _fit_composition_bins.function(
                self.score_model,
                self.exon,
                n_max=2,
                len_lo=30,
                len_hi=30,
                bin_width=1,
                per_bin=400,
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
            self._scores(residual, [bytes(60)])

    def test_forward_subtracts_a_linear_function_of_composition(self):
        residual = self._fit(len_lo=30, len_hi=31, bin_width=1)
        middles, _ = random_middles(256, 30, seed=1)
        subtracted = self._scores(self.score_model, middles) - self._scores(
            residual, middles
        )
        # what was subtracted is intercept + bow @ beta, i.e. an affine function of
        # the bag-of-k-mers features, so it fits with no residual.
        design = np.column_stack([np.ones(len(middles)), bow_features(middles, 2)])
        coef, *_ = np.linalg.lstsq(design, subtracted, rcond=None)
        np.testing.assert_allclose(design @ coef, subtracted, atol=1e-4)

    def test_fit_survives_state_dict_round_trip(self):
        residual = self._fit(len_lo=30, len_hi=31, bin_width=1)
        middles, _ = random_middles(8, 30, seed=2)
        before = self._scores(residual, middles)
        zeroed = _residual_module(
            self.score_model,
            self.exon,
            2,
            dict(
                edge0=0.0,
                step=1.0,
                # pylint: disable=protected-access
                intercepts=np.zeros_like(residual._intercepts.numpy()),
                betas=np.zeros_like(residual._betas.numpy()),
                r2s=np.zeros_like(residual.composition_r2s),
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
            fresh = [
                row.tobytes()
                for row in np.random.default_rng(length).integers(
                    0, 4, size=(256, length), dtype=np.uint8
                )
            ]
            self.assertAlmostEqual(
                oracle.membership_queries(fresh).mean(), 0.5, delta=0.2
            )


if __name__ == "__main__":
    unittest.main()
