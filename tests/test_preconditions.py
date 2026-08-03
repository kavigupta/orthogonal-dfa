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


# Counts 1s mod 9 and accepts one residue: strongly connected and learnable,
# but only ~13% of length-40 strings are accepted, so a balance requirement on
# the acceptance rate would reject it.
MOD9_SKEWED = DFA(
    states=set(range(9)),
    input_symbols={0, 1},
    transitions={q: {0: q, 1: (q + 1) % 9} for q in range(9)},
    initial_state=0,
    final_states={1},
    allow_partial=False,
)


# cp = 0.028: admitted at the 0.02 bar, rejected at the old 0.05 one.
MARGINAL_CLASS_PRESERVING = DFA(
    states=set(range(7)),
    input_symbols={0, 1},
    transitions={
        0: {0: 5, 1: 0},
        1: {0: 6, 1: 6},
        2: {0: 3, 1: 6},
        3: {0: 2, 1: 1},
        4: {0: 4, 1: 5},
        5: {0: 1, 1: 5},
        6: {0: 0, 1: 1},
    },
    initial_state=0,
    final_states={4, 6},
    allow_partial=False,
)

# cp = 0.011: below the bar, and E-L* does not learn it within a 420s budget.
TOO_FEW_CLASS_PRESERVING = DFA(
    states=set(range(5)),
    input_symbols={0, 1},
    transitions={
        0: {0: 3, 1: 1},
        1: {0: 1, 1: 2},
        2: {0: 3, 1: 2},
        3: {0: 0, 1: 3},
        4: {0: 4, 1: 2},
    },
    initial_state=0,
    final_states={1, 4},
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
        # The other two checks pass a constant language trivially, so the
        # acceptance rate is the only one that can reject it.
        for finals in ({0}, set()):
            report = P.satisfies_preconditions(_constant_dfa(finals), length=40)
            self.assertFalse(report)
            self.assertIn("acceptance rate", report.reasons[0])

    def test_acceptance_rate_band_matches_the_give_up_bound(self):
        # The band is give_up_check's assumption about its prefixes, so a target
        # skewed past it is rejected rather than run and given up on.
        skewed = P.satisfies_preconditions(
            MOD9_SKEWED, length=40, min_accept_or_reject=0.2
        )
        self.assertFalse(skewed)
        self.assertIn("acceptance rate", skewed.reasons[0])
        self.assertTrue(
            P.satisfies_preconditions(MOD9_SKEWED, length=40, min_accept_or_reject=0.02)
        )

    def test_skewed_but_non_degenerate_target_passes(self):
        # Only 13% accepted, so the acceptance-rate check must not impose
        # balance: E-L* learns this one to accuracy 1.0.
        self.assertLess(P.acceptance_rate(MOD9_SKEWED, length=40), 0.15)
        self.assertTrue(P.satisfies_preconditions(MOD9_SKEWED, length=40))

    def test_marginal_class_preserving_target_passes(self):
        # Between the old 0.05 bar and the current 0.02 one: E-L* learns these.
        cp = P.class_preserving_fraction(MARGINAL_CLASS_PRESERVING, length=40)
        self.assertTrue(0.02 <= cp < 0.05)
        self.assertTrue(P.satisfies_preconditions(MARGINAL_CLASS_PRESERVING, length=40))

    def test_too_few_class_preserving_suffixes_fails(self):
        report = P.satisfies_preconditions(TOO_FEW_CLASS_PRESERVING, length=40)
        self.assertFalse(report)
        self.assertIn("class-preserving", report.reasons[0])

    def test_ceiling_catches_transient_states(self):
        # Difficult09 is non-degenerate and class-preserving, so only the
        # covered-accuracy ceiling catches it: the decision lives in transient
        # states.
        self.assertNotIn(P.acceptance_rate(DIFFICULT09, length=40), (0.0, 1.0))
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
