import unittest

from orthogonal_dfa.l_star.statistics import (
    MINORITY_UNDECIDED_RATIO,
    evidence_margin_for_population_size,
    reads_minority_bounded,
)

#: `selection_not_minority_bounded` in proofs/OrthoDFA/OrthoDFA/Proofs/FamilyRead.lean:
#: at this center and signal the cross and FNR criteria alone pick (2, 7) over 7.
CENTER = 3 / 5
SIGNAL = 7 / 20
CROSS_LIMIT = 1e-3


class TestReadMinority(unittest.TestCase):
    def test_counterexample_band_is_not_bounded(self):
        self.assertFalse(
            reads_minority_bounded(
                2,
                7,
                7,
                accept_rate=CENTER + SIGNAL,
                reject_rate=CENTER - SIGNAL,
                ratio=MINORITY_UNDECIDED_RATIO,
            )
        )

    def test_selection_rejects_counterexample_band(self):
        self.assertIsNone(
            evidence_margin_for_population_size(
                SIGNAL, CROSS_LIMIT, 0.33, 7, center=CENTER
            )
        )

    def test_shipped_band_is_bounded(self):
        self.assertTrue(self._shipped_band(MINORITY_UNDECIDED_RATIO))

    def test_shipped_band_fails_a_tighter_ratio(self):
        # The state with half its suffixes in the language reads each way 4.1e-4 of
        # the time and undecided nearly always.
        self.assertFalse(self._shipped_band(4e-4))

    @staticmethod
    def _shipped_band(ratio):
        return reads_minority_bounded(
            20, 42, 62, accept_rate=0.8, reject_rate=0.2, ratio=ratio
        )
