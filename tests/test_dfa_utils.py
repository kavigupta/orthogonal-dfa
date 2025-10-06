import unittest

import numpy as np
from parameterized import parameterized

from orthogonal_dfa.manual_dfa.stop_codon_dfa import stop_codon_dfa
from orthogonal_dfa.utils.dfa import canonicalize_states, hash_dfa, rename_states


class TestCanonicalizeDFA(unittest.TestCase):
    @parameterized.expand([(seed,) for seed in range(1000)])
    def test_canonicalize_dfa(self, seed):
        dfa = stop_codon_dfa()
        rng = np.random.default_rng(seed)
        states = list(dfa.states)
        rng.shuffle(states)
        dfa_shuf = rename_states(dfa, {old: new for new, old in enumerate(states)})
        dfa_canon = canonicalize_states(dfa)
        dfa_shuf_canon = canonicalize_states(dfa_shuf)
        self.assertEqual(dfa_canon.states, dfa_shuf_canon.states)
        self.assertEqual(dfa_canon.accepting_states, dfa_shuf_canon.accepting_states)
        self.assertEqual(dfa_canon.initial_state, dfa_shuf_canon.initial_state)
        self.assertEqual(dfa_canon.alphabet, dfa_shuf_canon.alphabet)
        for state in dfa_canon.states:
            self.assertEqual(
                dfa_canon.transition_function[state],
                dfa_shuf_canon.transition_function[state],
            )
        self.assertEqual(hash_dfa(dfa_canon), hash_dfa(dfa_shuf_canon))
