import unittest

from orthogonal_dfa.l_star.statistics import (
    MINORITY_READ_LIMIT,
    MINORITY_UNDECIDED_RATIO,
    evidence_margin_for_population_size,
    population_size_and_evidence_margin,
    reads_minority_bounded,
)

#: `selection_not_trichotomy` and `selection_not_minority_bounded` in
#: proofs/OrthoDFA/OrthoDFA/Proofs/FamilyRead.lean: at each center and signal the cross
#: and FNR criteria alone pick a band that fails one half of the check.
TRICHOTOMY_CASE = dict(
    center=17 / 30, signal=17 / 30 - 67 / 500, cross_limit=(2 / 15) ** 15 * (1 + 1e-7)
)
RATIO_CASE = dict(center=3 / 5, signal=7 / 20, cross_limit=1e-3)


class TestReadMinority(unittest.TestCase):
    def test_trichotomy_counterexample_band_fails(self):
        self.assertFalse(self._passes(2, 15, 15, **TRICHOTOMY_CASE))

    def test_ratio_counterexample_band_fails(self):
        self.assertFalse(self._passes(2, 7, 7, **RATIO_CASE))

    def test_selection_rejects_counterexample_bands(self):
        for N, case in ((15, TRICHOTOMY_CASE), (7, RATIO_CASE)):
            self.assertIsNone(
                evidence_margin_for_population_size(
                    case["signal"], case["cross_limit"], 0.33, N, center=case["center"]
                )
            )

    def test_shipped_band_passes(self):
        self.assertTrue(
            self._shipped_band(MINORITY_READ_LIMIT, MINORITY_UNDECIDED_RATIO)
        )

    def test_shipped_band_fails_a_tighter_minority_limit(self):
        # The state with half its suffixes in the language reads each way 4.1e-4 of
        # the time, with neither side under the cross limit.
        self.assertFalse(self._shipped_band(1e-4, MINORITY_UNDECIDED_RATIO))

    def test_shipped_band_fails_a_tighter_ratio(self):
        # That state reads undecided nearly always.
        self.assertFalse(self._shipped_band(MINORITY_READ_LIMIT, 4e-4))

    def test_selection_unmoved_by_a_tiny_reject_rate(self):
        # A state whose suffixes all lead out reads undecided under float epsilon of
        # the time, and accept rarer still.  The decision boundary is clamped to the
        # signal, so the off-language rate can be zero or near it.
        # The sizes are those the cross and FNR criteria alone pick.
        for center, size in ((0.2, 126), (0.202, 122)):
            self.assertEqual(
                population_size_and_evidence_margin(0.2, 1.5e-7, 0.01, center=center)[
                    0
                ],
                size,
            )

    @staticmethod
    def _passes(k_low, k_high, N, *, center, signal, cross_limit):
        return reads_minority_bounded(
            k_low,
            k_high,
            N,
            accept_rate=center + signal,
            reject_rate=center - signal,
            limit=cross_limit,
            minority_limit=MINORITY_READ_LIMIT,
            ratio=MINORITY_UNDECIDED_RATIO,
        )

    @staticmethod
    def _shipped_band(minority_limit, ratio):
        return reads_minority_bounded(
            20,
            42,
            62,
            accept_rate=0.8,
            reject_rate=0.2,
            limit=1e-10,
            minority_limit=minority_limit,
            ratio=ratio,
        )
