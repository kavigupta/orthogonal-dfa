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

# The [336] false positive: every state is on a cycle (structurally recurrent),
# yet length-40 strings only ever end in {1,3,4} -- states 0 and 2 get zero
# prefix mass, so they are uncovered. The old infinite-reachability check
# admitted this (ceiling 1.0) but E-L* only reaches ~0.75; the covered-accuracy
# ceiling catches it. Guards against reintroducing the structural check.
RECURRENT_BUT_UNCOVERED = DFA(
    states={0, 1, 2, 3, 4},
    input_symbols={0, 1},
    transitions={
        0: {0: 3, 1: 2},
        1: {0: 3, 1: 4},
        2: {0: 0, 1: 0},
        3: {0: 1, 1: 4},
        4: {0: 4, 1: 1},
    },
    initial_state=0,
    final_states={4},
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


class TestCoverability(unittest.TestCase):
    def test_covered_states_strongly_connected(self):
        # MOD3 is strongly connected and balanced: every state is a length-40
        # endpoint often enough to be built.
        self.assertEqual(
            P.covered_states(MOD3, length=40, num_samples=500), set(MOD3.states)
        )

    def test_covered_states_excludes_transient(self):
        # DIFFICULT09: length-40 strings only ever reach the two absorbing sinks.
        self.assertEqual(
            P.covered_states(DIFFICULT09, length=40, num_samples=500), {3, 4}
        )

    def test_covered_states_excludes_recurrent_but_uncovered(self):
        # Structurally every state is on a cycle, but 0 and 2 get no length-40
        # prefix mass -- the distinction structural reachability would miss.
        self.assertEqual(
            P.covered_states(RECURRENT_BUT_UNCOVERED, length=40, num_samples=1000),
            {1, 3, 4},
        )


class TestSatisfiesPreconditions(unittest.TestCase):
    def test_balanced_target_passes(self):
        self.assertTrue(P.satisfies_preconditions(MOD3, length=40))

    def test_degenerate_acceptance_rate_fails(self):
        self.assertFalse(P.satisfies_preconditions(_constant_dfa({0}), length=40))
        self.assertFalse(P.satisfies_preconditions(_constant_dfa(set()), length=40))

    def test_ceiling_catches_transient_states(self):
        # Difficult09 is balanced and class-preserving, so only the covered-
        # accuracy ceiling catches it: the decision lives in transient states.
        self.assertTrue(0.15 <= P.acceptance_rate(DIFFICULT09, length=40) <= 0.85)
        self.assertGreaterEqual(
            P.class_preserving_fraction(DIFFICULT09, length=40), 0.05
        )
        self.assertLess(P.covered_accuracy_ceiling(DIFFICULT09, length=40), 0.99)
        self.assertFalse(P.satisfies_preconditions(DIFFICULT09, length=40))

    def test_ceiling_catches_recurrent_but_uncovered_states(self):
        # The [336] regression: structural reachability admitted it (all states
        # on cycles) but the decision lives in an uncovered state, so E-L* is
        # capped well below 1.0 and the ceiling must reject it.
        self.assertLess(
            P.covered_accuracy_ceiling(RECURRENT_BUT_UNCOVERED, length=40), 0.99
        )
        self.assertFalse(P.satisfies_preconditions(RECURRENT_BUT_UNCOVERED, length=40))


if __name__ == "__main__":
    unittest.main()
