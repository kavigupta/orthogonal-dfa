import unittest

import numpy as np
from automata.fa.dfa import DFA

from orthogonal_dfa.l_star.dfa_utils import held_out_phi
from tests.dfas import PARITY

_PARITY = DFA(
    states={0, 1},
    input_symbols={0, 1},
    transitions={0: {0: 0, 1: 1}, 1: {0: 1, 1: 0}},
    initial_state=0,
    final_states={1},
)


def _labelled(strings, labels):
    return [bytes(w) for w in strings], np.asarray(labels, dtype=float)


class TestHeldOutPhi(unittest.TestCase):
    def test_states_that_split_the_labels_score_one(self):
        fit = _labelled([[0], [1], [1, 1], [0, 1]], [0, 1, 0, 1])
        read = _labelled([[1, 0], [0, 0], [1, 1, 1]], [1, 0, 1])
        self.assertAlmostEqual(1.0, held_out_phi(_PARITY, fit, read))

    def test_states_unrelated_to_the_labels_score_zero(self):
        fit = _labelled([[0], [1], [0, 0], [0, 1]], [0, 0, 1, 1])
        read = _labelled([[1, 0], [0, 0]], [1, 0])
        self.assertEqual(0.0, held_out_phi(_PARITY, fit, read))

    def test_a_state_fit_never_reaches_takes_the_overall_rate(self):
        # Only state 0 is fit, at the overall rate, so no state accepts.
        fit = _labelled([[0], [0, 0]], [0, 1])
        read = _labelled([[1], [0]], [1, 0])
        self.assertEqual(0.0, held_out_phi(_PARITY, fit, read))

    def test_a_split_the_held_out_labels_invert_scores_minus_one(self):
        fit = _labelled([[0], [1]], [0, 1])
        read = _labelled([[0], [1], [0, 1]], [1, 0, 0])
        self.assertAlmostEqual(-1.0, held_out_phi(_PARITY, fit, read))

    def test_a_state_accepts_only_above_the_overall_rate(self):
        # The DFA accepts in state 0, but the fit puts state 1 above the overall rate.
        fit = _labelled(
            [[0], [0, 0], [0, 0, 0], [1], [1, 0], [0, 1]], [0, 0, 1, 1, 1, 0]
        )
        read = _labelled([[0], [1]], [0, 1])
        self.assertAlmostEqual(1.0, held_out_phi(PARITY, fit, read))


if __name__ == "__main__":
    unittest.main()
