import unittest

import numpy as np
from automata.fa.dfa import DFA
from parameterized import parameterized

from orthogonal_dfa.l_star.examples.benchmark_generator import (
    DFAOracle,
    build_star_l_star_dfa,
    sample_balanced_benchmark,
    sample_inner_dfa,
    sample_star_l_star,
)
from orthogonal_dfa.l_star.structures import SymmetricBernoulli
from tests.test_lstar import compute_dfa_accuracy, compute_dfa_for_oracle

# ===================================================================
# Inner DFA sampling
# ===================================================================


class TestSampleInnerDFA(unittest.TestCase):
    @parameterized.expand(
        [
            (4, 2, 0),
            (5, 2, 1),
            (5, 3, 0),
            (5, 3, 2),
            (6, 2, 0),
            (4, 4, 1),
        ]
    )
    def test_separator_holds(self, num_states, alphabet_size, sep_char):
        rng = np.random.default_rng(42)
        for _ in range(30):
            dfa = sample_inner_dfa(
                rng,
                num_states=num_states,
                alphabet_size=alphabet_size,
                separator_char=sep_char,
            )
            for q in dfa.states:
                self.assertNotIn(
                    dfa.transitions[q][sep_char],
                    dfa.final_states,
                    f"δ({q}, {sep_char}) ∈ F",
                )
            self.assertNotIn(dfa.initial_state, dfa.final_states)
            self.assertTrue(len(dfa.final_states) > 0)

    def test_deterministic(self):
        a = sample_inner_dfa(
            np.random.default_rng(0), num_states=5, alphabet_size=2, separator_char=0
        )
        b = sample_inner_dfa(
            np.random.default_rng(0), num_states=5, alphabet_size=2, separator_char=0
        )
        self.assertEqual(a, b)


# ===================================================================
# Subset construction (Σ*LΣ*)
# ===================================================================


def _brute_force_contains_substring_in_l(string, inner_dfa):
    n = len(string)
    for i in range(n + 1):
        for j in range(i, n + 1):
            if inner_dfa.accepts_input(string[i:j]):
                return True
    return False


class TestBuildStarLStarDFA(unittest.TestCase):
    @parameterized.expand([(seed,) for seed in range(20)])
    def test_matches_brute_force(self, seed):
        rng = np.random.default_rng(seed)
        inner = sample_inner_dfa(rng, num_states=4, alphabet_size=2, separator_char=0)
        outer = build_star_l_star_dfa(inner)
        test_rng = np.random.default_rng(seed + 1000)
        for _ in range(500):
            length = int(test_rng.integers(0, 18))
            s = test_rng.integers(0, 2, size=length).tolist()
            expected = _brute_force_contains_substring_in_l(s, inner)
            self.assertEqual(expected, outer.accepts_input(s))

    def test_epsilon_in_l(self):
        dfa = build_star_l_star_dfa(
            _trivial_dfa(accepting_initial=True, alphabet_size=2)
        )
        self.assertTrue(dfa.accepts_input([]))
        self.assertTrue(dfa.accepts_input([0, 1, 0]))

    def test_empty_l(self):
        dfa = build_star_l_star_dfa(
            _trivial_dfa(accepting_initial=False, alphabet_size=2, make_empty=True)
        )
        self.assertFalse(dfa.accepts_input([]))
        self.assertFalse(dfa.accepts_input([0, 1, 0]))


# ===================================================================
# End-to-end: sample_star_l_star + DFAOracle
# ===================================================================


class TestSampleStarLStar(unittest.TestCase):
    @parameterized.expand([(seed,) for seed in range(10)])
    def test_round_trip(self, seed):
        rng = np.random.default_rng(seed)
        outer, inner, sep_char = sample_star_l_star(rng, alphabet_size=2)
        for q in inner.states:
            self.assertNotIn(inner.transitions[q][sep_char], inner.final_states)
        test_rng = np.random.default_rng(seed + 5000)
        for _ in range(300):
            length = int(test_rng.integers(0, 16))
            s = test_rng.integers(0, 2, size=length).tolist()
            expected = _brute_force_contains_substring_in_l(s, inner)
            self.assertEqual(expected, outer.accepts_input(s))

    def test_nontrivial(self):
        rng = np.random.default_rng(77)
        outer, _, _ = sample_star_l_star(rng, alphabet_size=2)
        self.assertGreaterEqual(len(outer.states), 2)
        self.assertGreaterEqual(len(outer.final_states), 1)


class TestDFAOracle(unittest.TestCase):
    def test_noiseless_matches_dfa(self):
        rng = np.random.default_rng(0)
        outer, _, _ = sample_star_l_star(rng, alphabet_size=2)
        oracle = DFAOracle(SymmetricBernoulli(p_correct=1.0), seed=0, dfa=outer)
        self.assertEqual(oracle.alphabet_size, 2)
        test_rng = np.random.default_rng(1)
        for _ in range(500):
            s = test_rng.integers(0, 2, size=int(test_rng.integers(0, 20))).tolist()
            self.assertEqual(oracle.membership_query(s), outer.accepts_input(s))


# ===================================================================
# Noisy L* on generated benchmarks
# ===================================================================


benchmark_allowed_error = 0.05


class TestLStarOnGeneratedBenchmarks(unittest.TestCase):
    @parameterized.expand([(seed,) for seed in range(3)])
    def test_generated_benchmark(self, seed):
        outer, _, _ = sample_balanced_benchmark(
            seed,
            alphabet_size=2,
            num_inner_states=12,
            num_outer_states=10,
            probe_length=40,
            min_accept_or_reject=0.15,
        )
        print(outer)
        oracle_creator = lambda nm, s, _dfa=outer: DFAOracle(nm, s, _dfa)
        _, dfa, _ = compute_dfa_for_oracle(
            oracle_creator, min_signal_strength=0.3, seed=0
        )
        accuracy, fp, fn = compute_dfa_accuracy(dfa, oracle_creator)
        if accuracy < 1 - benchmark_allowed_error:
            self.fail(
                f"DFA incorrect (accuracy {accuracy:.3f}). "
                f"FP: {len(fp)}, FN: {len(fn)}"
            )


# ===================================================================
# Helpers
# ===================================================================


def _trivial_dfa(*, accepting_initial, alphabet_size, make_empty=False):
    if make_empty:
        return DFA(
            states={0},
            input_symbols=set(range(alphabet_size)),
            transitions={0: {c: 0 for c in range(alphabet_size)}},
            initial_state=0,
            final_states=set(),
        )
    return DFA(
        states={0, 1},
        input_symbols=set(range(alphabet_size)),
        transitions={
            0: {c: 1 for c in range(alphabet_size)},
            1: {c: 1 for c in range(alphabet_size)},
        },
        initial_state=0,
        final_states={0} if accepting_initial else {1},
    )
