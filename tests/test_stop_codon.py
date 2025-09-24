import unittest

import frame_alignment_checks as fac
import numpy as np

from orthogonal_dfa.manual_dfa.stop_codon_dfa import stop_codon_dfa


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
