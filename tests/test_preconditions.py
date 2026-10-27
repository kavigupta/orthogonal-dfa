# pylint: disable=duplicate-code
import unittest
from dataclasses import dataclass

from automata.fa.dfa import DFA
from parameterized import parameterized

from orthogonal_dfa.l_star import preconditions as P
from orthogonal_dfa.l_star.sampler import Sampler, UniformSampler

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
# admitted this (ceiling 1.0) but E-L* only reaches ~0.75; the non-start-zone accuracy
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


def _thin_pockets(alphabet, pockets, depth):
    """Accepts on the parity of symbols below alphabet/2 and of pocket entries.

    The pair (z_i, z_i) enters pocket i, ``depth`` states in which z_i restarts
    the pocket, after which the run returns to the core.
    """
    zs = list(range(alphabet - pockets, alphabet))
    kinds = ["core"] + [("arm", i) for i in range(pockets)]
    kinds += [("pocket", i, j) for i in range(pockets) for j in range(depth)]
    states = [(bit, kind) for bit in (0, 1) for kind in kinds]

    def step(state, x):
        bit, kind = state
        bit ^= x < alphabet // 2
        if kind[0] != "pocket":
            if x not in zs:
                return bit, "core"
            i = zs.index(x)
            return (
                (1 - bit, ("pocket", i, 0)) if kind == ("arm", i) else (bit, ("arm", i))
            )
        _, i, j = kind
        if x == zs[i]:
            return bit, ("pocket", i, 0)
        return (bit, ("pocket", i, j + 1)) if j + 1 < depth else (bit, "core")

    return DFA(
        states=set(states),
        input_symbols=set(range(alphabet)),
        transitions={q: {x: step(q, x) for x in range(alphabet)} for q in states},
        initial_state=(0, "core"),
        final_states={q for q in states if q[0]},
        allow_partial=False,
    ).minify()


# Every pocket state is uncovered, but whole runs land in it at about the window's
# per-position rate, so it is outside the start zone.
THIN_POCKETS = _thin_pockets(alphabet=48, pockets=3, depth=2)


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


class TestStarts(unittest.TestCase):
    def _starts(self, dfa, num_samples):
        return set(P.non_start_zone_states(dfa, length=40, num_samples=num_samples))

    def test_starts_strongly_connected(self):
        self.assertEqual(self._starts(MOD3, 500), set(MOD3.states))

    def test_starts_exclude_transient(self):
        # DIFFICULT09: length-40 strings only ever reach the two absorbing sinks.
        self.assertEqual(self._starts(DIFFICULT09, 500), {3, 4})


class TestSatisfiesPreconditions(unittest.TestCase):
    def test_balanced_target_passes(self):
        self.assertTrue(P.satisfies_preconditions(MOD3, length=40))

    def test_degenerate_acceptance_rate_fails(self):
        # The other two checks pass a constant language trivially, so the
        # acceptance rate is the only one that can reject it.
        for finals in ({0}, set()):
            report = P.satisfies_preconditions(_constant_dfa(finals), length=40)
            self.assertFalse(report)
            self.assertIn("degenerate", report.reasons[0])

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
        # non-start-zone accuracy ceiling catches it: the decision lives in transient
        # states.
        self.assertNotIn(P.acceptance_rate(DIFFICULT09, length=40), (0.0, 1.0))
        self.assertGreaterEqual(
            P.class_preserving_fraction(DIFFICULT09, length=40), 0.05
        )
        self.assertLess(P.non_start_zone_accuracy_ceiling(DIFFICULT09, length=40), 0.99)
        self.assertFalse(P.satisfies_preconditions(DIFFICULT09, length=40))

    def test_ceiling_catches_recurrent_but_uncovered_states(self):
        # The [336] regression: structural reachability admitted it (all states
        # on cycles) but the decision lives in an uncovered state, so E-L* is
        # capped well below 1.0 and the ceiling must reject it.
        self.assertLess(
            P.non_start_zone_accuracy_ceiling(RECURRENT_BUT_UNCOVERED, length=40), 0.99
        )
        self.assertFalse(P.satisfies_preconditions(RECURRENT_BUT_UNCOVERED, length=40))

    def test_thinly_sampled_but_non_start_zone_states_pass(self):
        report = P.satisfies_preconditions(THIN_POCKETS, length=40, short_circuit=False)
        self.assertTrue(report, report.reasons)


class TestRelativeRule(unittest.TestCase):
    def test_zone_contains_start(self):
        zones = P.non_start_zone_states(THIN_POCKETS, length=40)
        for q, zone in zones.items():
            self.assertIn(q, zone)


class _ZeroRunSampler(Sampler):
    """Emits ``0`` everywhere, so MOD3 never leaves its initial state."""

    def __init__(self, length: int):
        self.length = length

    def sample(self, rng, alphabet_size):
        return bytes(self.length)

    def symbol_weights(self, alphabet_size):
        return [1] + [0] * (alphabet_size - 1)


class TestSampler(unittest.TestCase):
    """The conditions are properties of the target under a sampling distribution,
    so a caller learning over a different one has to measure over that one."""

    def test_verdict_follows_the_sampler(self):
        self.assertTrue(P.satisfies_preconditions(MOD3, length=40))
        report = P.satisfies_preconditions(MOD3, length=40, sampler=_ZeroRunSampler(40))
        self.assertFalse(report)
        self.assertEqual(report.acceptance_rate, 0.0)

    def test_default_is_the_uniform_sampler(self):
        self.assertEqual(
            P.acceptance_rate(MOD3, length=40, sampler=UniformSampler(40)),
            P.acceptance_rate(MOD3, length=40),
        )

    def test_length_mismatch_asserts(self):
        with self.assertRaises(AssertionError):
            P.satisfies_preconditions(MOD3, length=40, sampler=UniformSampler(15))


@dataclass(frozen=True)
class _SampledUniform(UniformSampler):
    """Uniform, but not by type, so the conditions are sampled rather than exact."""


class TestExactMatchesSampled(unittest.TestCase):
    @parameterized.expand(
        [
            ("mod3", MOD3),
            ("difficult09", DIFFICULT09),
            ("recurrent_but_uncovered", RECURRENT_BUT_UNCOVERED),
            ("mod9_skewed", MOD9_SKEWED),
            ("marginal_cp", MARGINAL_CLASS_PRESERVING),
            ("too_few_cp", TOO_FEW_CLASS_PRESERVING),
        ]
    )
    def test_exact_matches_sampled(self, _name, dfa):
        exact = P.satisfies_preconditions(dfa, length=40, short_circuit=False)
        sampled = P.satisfies_preconditions(
            dfa, length=40, short_circuit=False, sampler=_SampledUniform(40)
        )
        self.assertEqual(bool(exact), bool(sampled))
        self.assertEqual(exact.unvisited_states, sampled.unvisited_states)
        for field in (
            "acceptance_rate",
            "class_preserving_fraction",
            "non_start_zone_accuracy_ceiling",
        ):
            # 4 standard deviations of a proportion over 2000 samples
            self.assertAlmostEqual(
                getattr(exact, field), getattr(sampled, field), delta=0.045
            )


if __name__ == "__main__":
    unittest.main()
