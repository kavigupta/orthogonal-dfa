import unittest

from automata.fa.dfa import DFA

from orthogonal_dfa.l_star import preconditions as P

# mod-3 counter on 1s: strongly connected, balanced under uniform sampling.
MOD3 = DFA(
    states={0, 1, 2},
    input_symbols={0, 1},
    transitions={0: {0: 0, 1: 1}, 1: {0: 1, 1: 2}, 2: {0: 2, 1: 0}},
    initial_state=0,
    final_states={1},
    allow_partial=False,
)


def _constant_dfa(final_states):
    return DFA(
        states={0},
        input_symbols={0, 1},
        transitions={0: {0: 0, 1: 0}},
        initial_state=0,
        final_states=final_states,
        allow_partial=False,
    )


class TestMeasures(unittest.TestCase):
    def test_acceptance_rate_degenerate_languages(self):
        self.assertEqual(
            P.acceptance_rate(_constant_dfa({0}), length=40, num_samples=200), 1.0
        )
        self.assertEqual(
            P.acceptance_rate(_constant_dfa(set()), length=40, num_samples=200), 0.0
        )

    def test_class_preserving_fraction_in_unit_interval(self):
        frac = P.class_preserving_fraction(MOD3, length=40, num_samples=500)
        self.assertTrue(0.0 <= frac <= 1.0)


class TestSatisfiesPreconditions(unittest.TestCase):
    def test_balanced_target_passes(self):
        self.assertTrue(P.satisfies_preconditions(MOD3, length=40))

    def test_degenerate_acceptance_rate_fails(self):
        self.assertFalse(P.satisfies_preconditions(_constant_dfa({0}), length=40))
        self.assertFalse(P.satisfies_preconditions(_constant_dfa(set()), length=40))


if __name__ == "__main__":
    unittest.main()
