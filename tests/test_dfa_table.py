import unittest

import numpy as np
from parameterized import parameterized

from orthogonal_dfa.l_star.dfa_utils import (
    dfa_table,
    states_intermediate,
    table_end_states,
)
from orthogonal_dfa.l_star.examples.random_dfa import sample_random_dfa
from tests.dfas import PARITY


def _random_dfa(seed):
    return sample_random_dfa(
        np.random.default_rng(seed), num_states=12, alphabet_size=3
    )


class TestDFATable(unittest.TestCase):
    def test_start_and_accepting_of_parity(self):
        table = dfa_table(PARITY)
        self.assertEqual(table["transitions"], [[0, 1], [1, 0]])
        self.assertEqual(table["start"], 0)
        self.assertEqual(table["accepting"], [1, 0])

    @parameterized.expand([(seed,) for seed in range(20)])
    def test_start_and_accepting_match_the_dfa(self, seed):
        dfa = _random_dfa(seed)
        table = dfa_table(dfa)
        states = sorted(dfa.states, key=str)
        self.assertEqual(states[table["start"]], dfa.initial_state)
        self.assertEqual(
            {states[i] for i, a in enumerate(table["accepting"]) if a},
            dfa.final_states,
        )

    @parameterized.expand(
        [("parity", PARITY, 2)] + [(f"random{s}", _random_dfa(s), 3) for s in range(20)]
    )
    def test_end_states_match_the_dfa(self, _, dfa, alphabet_size):
        rng = np.random.default_rng(0)
        table = dfa_table(dfa)
        states = sorted(dfa.states, key=str)
        for length in (0, 1, 5, 12):
            strings = rng.integers(alphabet_size, size=(50, length))
            ends = table_end_states(
                np.array(table["transitions"]), table["start"], strings
            )
            for w, end in zip(strings, ends):
                expected = states_intermediate(dfa.initial_state, w, dfa)[-1]
                self.assertEqual(states[end], expected)
                self.assertEqual(
                    bool(table["accepting"][end]), dfa.accepts_input(w.tolist())
                )
