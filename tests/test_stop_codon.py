import unittest

import frame_alignment_checks as fac
import numpy as np

from orthogonal_dfa.manual_dfa.stop_codon_dfa import stop_codon_dfa
from orthogonal_dfa.utils.dfa import hash_dfa


class TestStopCodonDFA(unittest.TestCase):
    def test_stop_codon_dfa_on_examples(self):

        dfa = stop_codon_dfa()
        self.assertTrue(dfa.accepts("TAGCTAGATAG"))  # 0, 1, and 2
        self.assertTrue(dfa.accepts("TAGCTAAATGA"))  # 0, 1, and 2
        self.assertTrue(not dfa.accepts("TAGCTAGTAG"))  # 0, 1, and 1

    def test_on_random_examples(self):
        dfa = stop_codon_dfa()
        rng = np.random.RandomState(0)
        sequences = rng.choice(4, size=(10000, 120))
        is_stop_actual = fac.all_frames_closed(np.eye(4)[sequences])
        sequences = ["".join(x) for x in np.array(list("ACGT"))[sequences]]
        bad = [dfa.accepts(x) for x in sequences] != is_stop_actual
        self.assertTrue(not any(bad))

    def test_hash_regression_no_orf(self):
        self.assertEqual(
            hash_dfa(stop_codon_dfa()),
            "792296689170f7db5c7627bbcb44b9b62d24b48dc387f2d74b3d99c41c5204e0",
        )

    def test_hash_regression_no_orf_ta(self):
        self.assertEqual(
            hash_dfa(stop_codon_dfa(("TAA", "TGA"))),
            "98f344f8393d34154ada1870e27dda379401329ce02fe79da7dd3e698f82f65a",
        )

    def test_hash_regression_no_orf_phase_agnostic(self):
        self.assertEqual(
            hash_dfa(stop_codon_dfa(phase_agnostic=True)),
            "12263525fdd404c9fdb3b61446d9a2e7747b518e7294fe65562c6b7382f5f05d",
        )

    def test_hash_regression_no_orf_ta_phase_agnostic(self):
        self.assertEqual(
            hash_dfa(stop_codon_dfa(("TAA", "TGA"), phase_agnostic=True)),
            "03e6b9c5e915895975cb706b33ef258ce1d7c0bc75ff20c83eac18554170df68",
        )
