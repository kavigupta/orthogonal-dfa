# pylint: disable=duplicate-code
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

# {w : |w| >= 3 and w[2] == 0}: states 0,1,2 are transient (issue #128), 3/4 are
# absorbing sinks. No fixed-length string ends in 0/1/2.
DIFFICULT09 = DFA(
    states={0, 1, 2, 3, 4},
    input_symbols={0, 1},
    transitions={
        0: {0: 1, 1: 1},
        1: {0: 2, 1: 2},
        2: {0: 3, 1: 4},
        3: {0: 3, 1: 3},
        4: {0: 4, 1: 4},
    },
    initial_state=0,
    final_states={3},
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


class TestReachability(unittest.TestCase):
    def test_infinitely_reachable_states(self):
        # MOD3 is strongly connected: every state lies on a cycle.
        self.assertEqual(P.infinitely_reachable_states(MOD3), set(MOD3.states))
        # DIFFICULT09: only the two absorbing sinks; 0/1/2 are transient.
        self.assertEqual(P.infinitely_reachable_states(DIFFICULT09), {3, 4})

    def test_start_state_is_exempt(self):
        # A start reached only by the empty string (nothing cycles back) is not
        # infinitely reachable, but must not fail the precondition.
        dfa = DFA(
            states={0, 1},
            input_symbols={0, 1},
            transitions={0: {0: 1, 1: 1}, 1: {0: 1, 1: 1}},
            initial_state=0,
            final_states={1},
            allow_partial=False,
        )
        self.assertEqual(P.infinitely_reachable_states(dfa), {1})


class TestSatisfiesPreconditions(unittest.TestCase):
    def test_balanced_target_passes(self):
        self.assertTrue(P.satisfies_preconditions(MOD3, length=40))

    def test_degenerate_acceptance_rate_fails(self):
        self.assertFalse(P.satisfies_preconditions(_constant_dfa({0}), length=40))
        self.assertFalse(P.satisfies_preconditions(_constant_dfa(set()), length=40))

    def test_reachability_catches_transient_states(self):
        # Difficult09 is balanced and class-preserving, so only the reachability
        # condition catches it: non-start states 1/2 are not infinitely reachable.
        self.assertTrue(0.15 <= P.acceptance_rate(DIFFICULT09, length=40) <= 0.85)
        self.assertGreaterEqual(
            P.class_preserving_fraction(DIFFICULT09, length=40), 0.05
        )
        self.assertFalse(P.satisfies_preconditions(DIFFICULT09, length=40))


if __name__ == "__main__":
    unittest.main()
