import unittest

import numpy as np
from automata.fa.dfa import DFA

from orthogonal_dfa.analysis.elstar_learnability import _agreement, _relabeled_agreement


def two_sink_dfa():
    """State 0 rejects everything, state 1 accepts everything (both absorbing)."""
    return DFA(
        states={0, 1},
        input_symbols={0, 1},
        transitions={0: {0: 0, 1: 0}, 1: {0: 1, 1: 1}},
        initial_state=0,
        final_states={1},
    )


def first_symbol_dfa():
    """Routes by first symbol: 0 -> state 1, 1 -> state 2 (then absorbing). Labels
    accept iff the first symbol was 0."""
    return DFA(
        states={0, 1, 2},
        input_symbols={0, 1},
        transitions={0: {0: 1, 1: 2}, 1: {0: 1, 1: 1}, 2: {0: 2, 1: 2}},
        initial_state=0,
        final_states={1},
    )


class TestAgreement(unittest.TestCase):
    def setUp(self):
        self.dfa = two_sink_dfa()
        self.strings = [[0], [1], [0, 1], [1, 0]]
        self.truth = np.array([True, True, True, False])  # 3/4 accept

    def test_from_initial_state_rejects_everything(self):
        self.assertAlmostEqual(_agreement(self.dfa, self.strings, self.truth), 1 / 4)

    def test_agreement_from_a_chosen_start(self):
        self.assertAlmostEqual(
            _agreement(self.dfa, self.strings, self.truth, start=1), 3 / 4
        )


class TestRelabel(unittest.TestCase):
    def test_relabel_recovers_a_rule_the_labels_get_backwards(self):
        dfa = first_symbol_dfa()  # labels accept-iff-first-symbol-0
        strings = [[0, 1, 1], [0, 0], [1, 0], [1, 1, 0], [0, 1], [1, 0, 1]]
        truth = np.array([s[0] == 1 for s in strings])  # the OPPOSITE rule
        # E-L*'s own labels are exactly backwards here -> zero agreement...
        self.assertEqual(_agreement(dfa, strings, truth), 0.0)
        # ...but refitting each state's label recovers the routing perfectly.
        self.assertEqual(
            _relabeled_agreement(
                dfa, strings, truth, strings, truth, start=dfa.initial_state
            ),
            1.0,
        )

    def test_relabel_cannot_beat_a_collapsing_structure(self):
        # both strings-classes land in one absorbing sink -> relabel = majority class.
        dfa = two_sink_dfa()
        strings = [[0], [1], [0, 1], [1, 0]]
        truth = np.array([True, True, True, False])
        self.assertAlmostEqual(
            _relabeled_agreement(
                dfa, strings, truth, strings, truth, start=dfa.initial_state
            ),
            3 / 4,  # all collapse to state 0 -> its majority (accept) -> base rate
        )


if __name__ == "__main__":
    unittest.main()
