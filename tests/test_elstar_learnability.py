import unittest

import numpy as np
from automata.fa.dfa import DFA

from orthogonal_dfa.analysis.elstar_learnability import (
    _agreement,
    _best_reroot_agreement,
)


def two_sink_dfa():
    """State 0 rejects everything, state 1 accepts everything (both absorbing)."""
    return DFA(
        states={0, 1},
        input_symbols={0, 1},
        transitions={0: {0: 0, 1: 0}, 1: {0: 1, 1: 1}},
        initial_state=0,
        final_states={1},
    )


class TestAgreement(unittest.TestCase):
    def setUp(self):
        self.dfa = two_sink_dfa()
        self.strings = [[0], [1], [0, 1], [1, 0]]
        self.truth = np.array([True, True, True, False])  # 3/4 accept

    def test_from_initial_state_rejects_everything(self):
        # initial state 0 rejects all -> matches only the one False.
        self.assertAlmostEqual(_agreement(self.dfa, self.strings, self.truth), 1 / 4)

    def test_agreement_from_a_chosen_start(self):
        # state 1 accepts all -> matches the three Trues.
        self.assertAlmostEqual(
            _agreement(self.dfa, self.strings, self.truth, start=1), 3 / 4
        )

    def test_reroot_picks_the_better_start(self):
        # re-rooting at the accepting sink beats the initial reject sink.
        got = _best_reroot_agreement(
            self.dfa, self.strings, self.truth, self.strings, self.truth
        )
        self.assertAlmostEqual(got, 3 / 4)


if __name__ == "__main__":
    unittest.main()
