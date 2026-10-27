import collections
import itertools
import unittest

import numpy as np
from parameterized import parameterized

from orthogonal_dfa.l_star import preconditions as P
from orthogonal_dfa.l_star.examples.random_dfa import (
    _class_pair_bound,
    _class_preserving_fraction,
    _some_string_preserves,
    dfa_from_table,
    sample_accessible_tables,
    sample_minimal_admitted_dfa,
)


class TestSampleAccessibleTables(unittest.TestCase):
    def test_uniform_over_accessible_tables(self):
        # one symbol on 3 states: 0 -> a -> b -> anything, for a, b = 1, 2 or 2, 1
        expected = {
            (a, b, x) if a == 1 else (a, x, b)
            for a, b in itertools.permutations([1, 2])
            for x in range(3)
        }
        tables = sample_accessible_tables(
            np.random.default_rng(0), num_states=3, alphabet_size=1, count=6000
        )
        counts = collections.Counter(tuple(t[:, 0]) for t in tables)
        self.assertEqual(set(counts), expected)
        for count in counts.values():
            # 5 standard deviations of a count with mean 1000
            self.assertAlmostEqual(count, 1000, delta=5 * 30)


class TestClassPreservingChecks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.tables = sample_accessible_tables(
            rng, num_states=6, alphabet_size=2, count=300
        )
        self.finals = rng.random((300, 6)) < 0.5
        self.exact = [
            P.class_preserving_fraction(dfa_from_table(table, final), length=40)
            for table, final in zip(self.tables, self.finals)
        ]

    def test_pair_bound_bounds(self):
        bound = _class_pair_bound(self.tables, self.finals, 40)
        for b, exact in zip(bound, self.exact):
            self.assertLessEqual(exact, b + 1e-12)

    def test_dense_chain_is_exact(self):
        for table, final, exact in zip(self.tables, self.finals, self.exact):
            self.assertAlmostEqual(
                _class_preserving_fraction(table, final, 40), exact, delta=1e-12
            )

    def test_screen_passes_exactly_the_preserved(self):
        screened = _some_string_preserves(
            np.random.default_rng(2), self.tables, self.finals, length=40
        )
        for passed, exact in zip(screened, self.exact):
            if exact == 0:
                self.assertFalse(passed)
            if exact >= P.MIN_CLASS_PRESERVING_FRAC:
                self.assertTrue(passed)


class TestSampleMinimalAdmittedDFA(unittest.TestCase):
    @parameterized.expand([(3,), (5,)])
    def test_minimal_of_the_size_and_admitted(self, num_states):
        for i in range(3):
            aut = sample_minimal_admitted_dfa(
                np.random.default_rng((num_states, i)),
                num_states=num_states,
                alphabet_size=2,
                length=40,
            )
            self.assertEqual(len(aut.states), num_states)
            self.assertEqual(len(aut.minify().states), num_states)
            self.assertTrue(P.satisfies_preconditions(aut, length=40))


if __name__ == "__main__":
    unittest.main()
