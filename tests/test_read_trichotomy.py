import unittest

from orthogonal_dfa.l_star.statistics import (
    evidence_margin_for_population_size,
    reads_trichotomous,
)

#: `selection_not_trichotomy` in proofs/OrthoDFA/OrthoDFA/Proofs/FamilyRead.lean: at
#: this center and signal the cross and FNR criteria alone pick (2, 15) over 15.
CENTER = 17 / 30
SIGNAL = 17 / 30 - 67 / 500
CROSS_LIMIT = (2 / 15) ** 15 * (1 + 1e-7)


class TestReadTrichotomy(unittest.TestCase):
    def test_counterexample_band_is_not_trichotomous(self):
        self.assertFalse(
            reads_trichotomous(
                2,
                15,
                15,
                accept_rate=CENTER + SIGNAL,
                reject_rate=CENTER - SIGNAL,
                limit=CROSS_LIMIT,
            )
        )

    def test_selection_rejects_counterexample_band(self):
        self.assertIsNone(
            evidence_margin_for_population_size(
                SIGNAL, CROSS_LIMIT, 0.33, 15, center=CENTER
            )
        )

    def test_shipped_band_is_trichotomous(self):
        self.assertTrue(
            reads_trichotomous(
                20, 42, 62, accept_rate=0.8, reject_rate=0.2, limit=1e-10
            )
        )
