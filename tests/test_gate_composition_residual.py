import unittest

import numpy as np
from permacache import no_cache_global

from orthogonal_dfa.l_star.examples.composition_residual import bow_features
from orthogonal_dfa.l_star.examples.gate_composition_residual import (
    GateCompositionResidualScore,
    _fit_gate_bins,
    fit_gate_composition_residual,
    gate_residual_oracle,
)
from orthogonal_dfa.l_star.examples.spliceai_oracle import flanks, run_over_middles
from orthogonal_dfa.spliceai.exon_score import SpliceAIExonScore
from tests.spliceai_small import QUERY_LENGTH, random_middles, small_module_and_exon

# small monotonic fit so the tests stay fast on CPU
FAST = dict(n_max=2, per_bin=400, epochs=30, device="cpu", chunk=64)


class TestGateCompositionResidualScore(unittest.TestCase):
    def setUp(self):
        self.module, self.exon = small_module_and_exon()
        self.score_model = SpliceAIExonScore(self.module).eval()
        self.flank_l, self.flank_r = flanks(self.exon)

    def _fit(self, **band):
        # See the fit as the math currently is, not a stale permacache entry.
        with no_cache_global():
            return fit_gate_composition_residual(
                self.score_model, self.exon, **FAST, **band
            )

    def _scores(self, model, middles):
        return run_over_middles(
            model, self.flank_l, self.flank_r, middles, device="cpu", chunk=64
        )

    def _pred_lin(self, residual, middles):
        # The single-bin linear composition index the gate warps: intercept + bow @ beta.
        beta = residual._betas[0].numpy()  # pylint: disable=protected-access
        intercept = float(residual._intercepts[0])  # pylint: disable=protected-access
        return intercept + bow_features(middles, FAST["n_max"]) @ beta

    def test_single_bin_scores_without_nan(self):
        residual = self._fit(len_lo=30, len_hi=31, bin_width=1)
        # pylint: disable=protected-access
        self.assertEqual(residual._betas.shape[0], 1)
        self.assertEqual(len(residual.monotonics), 1)
        scored = self._scores(residual, [b"\x00\x01\x02" * 10])
        self.assertEqual(scored.shape, (1,))
        self.assertFalse(np.isnan(scored).any())

    def test_forward_subtracts_a_monotonic_function_of_composition(self):
        # residual = raw - monotonic(pred_lin), so what is subtracted (raw - residual)
        # is EXACTLY monotonic(pred_lin): a non-decreasing function of the linear
        # composition index.  (Contrast composition_residual, which subtracts a *linear*
        # function.)  In one bin, sorting middles by pred_lin makes subtracted monotone.
        residual = self._fit(len_lo=30, len_hi=31, bin_width=1)
        middles, _ = random_middles(256, 30, seed=1)
        subtracted = self._scores(self.score_model, middles) - self._scores(
            residual, middles
        )
        order = np.argsort(self._pred_lin(residual, middles))
        diffs = np.diff(subtracted[order])
        # monotone non-decreasing up to float error
        self.assertGreaterEqual(
            diffs.min(), -1e-3, "subtracted not monotone in pred_lin"
        )
        # and genuinely varying (the gate actually subtracts something)
        self.assertGreater(subtracted.std(), 0)

    def test_fit_survives_state_dict_round_trip(self):
        residual = self._fit(len_lo=30, len_hi=31, bin_width=1)
        middles = [
            row.tobytes()
            for row in np.random.default_rng(3).integers(
                0, 4, size=(8, 30), dtype=np.uint8
            )
        ]
        before = self._scores(residual, middles)
        # a fresh module with the same architecture but re-initialised monotonics
        fresh = GateCompositionResidualScore(
            self.score_model,
            flank_l_len=len(self.flank_l),
            n_max=FAST["n_max"],
            # pylint: disable=protected-access
            edge0=float(residual._edge0),
            step=float(residual._step),
            intercepts=residual._intercepts.numpy(),
            betas=residual._betas.numpy(),
            r2s=np.zeros(residual._betas.shape[0], np.float32),
            monotonics=[m.state_dict() for m in residual.monotonics],
        ).eval()
        np.testing.assert_allclose(self._scores(fresh, middles), before, rtol=1e-5)

    def test_oracle_is_balanced(self):
        with no_cache_global():
            oracle = gate_residual_oracle(
                self.exon,
                self.module,
                length=QUERY_LENGTH,
                len_lo=25,
                len_hi=40,
                bin_width=5,
                **FAST,
            )
        fresh = [
            row.tobytes()
            for row in np.random.default_rng(7).integers(
                0, 4, size=(256, QUERY_LENGTH), dtype=np.uint8
            )
        ]
        self.assertAlmostEqual(oracle.membership_queries(fresh).mean(), 0.5, delta=0.2)

    def test_fit_gate_bins_augments_the_linear_fit_with_monotonics(self):
        with no_cache_global():
            fit = _fit_gate_bins.function(
                self.score_model,
                self.exon,
                len_lo=25,
                len_hi=40,
                bin_width=5,
                **FAST,
            )
        n_bins = len(fit["betas"])
        self.assertEqual(len(fit["monotonics"]), n_bins)
        for key in ("edge0", "step", "intercepts", "betas", "r2s"):
            self.assertIn(key, fit)


if __name__ == "__main__":
    unittest.main()
