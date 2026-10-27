import collections
import itertools
import unittest

import numpy as np
from parameterized import parameterized

from orthogonal_dfa.l_star import preconditions as P
from orthogonal_dfa.l_star.examples.random_dfa import (
    _class_preserving_fraction,
    _is_minimal,
    accessible_tables,
    dfa_from_table,
    sample_minimal_admitted_dfa,
)


class TestAccessibleTables(unittest.TestCase):
    def test_uniform_over_accessible_tables(self):
        # one symbol on 3 states: 0 -> a -> b -> anything, for a, b = 1, 2 or 2, 1
        expected = {
            (a, b, x) if a == 1 else (a, x, b)
            for a, b in itertools.permutations([1, 2])
            for x in range(3)
        }
        counts = collections.Counter(
            tuple(t[:, 0]) for t in accessible_tables(0, 3, 1, 6000)
        )
        self.assertEqual(set(counts), expected)
        for count in counts.values():
            # 5 standard deviations of a count with mean 1000
            self.assertAlmostEqual(count, 1000, delta=5 * 30)


class TestCandidateChecks(unittest.TestCase):
    @parameterized.expand([(6,), (10,)])
    def test_match_automata_and_preconditions(self, num_states):
        rng = np.random.default_rng(num_states)
        tables = accessible_tables(num_states, num_states, 2, 200)
        for table, final_mask in zip(tables, rng.integers(1 << num_states, size=200)):
            aut = dfa_from_table(table, int(final_mask))
            self.assertEqual(
                _is_minimal(table, int(final_mask)),
                len(aut.minify().states) == num_states,
            )
            self.assertAlmostEqual(
                _class_preserving_fraction(table, int(final_mask), 40, -1.0),
                P.class_preserving_fraction(aut, length=40),
                delta=1e-12,
            )

    def test_early_exit_only_below_the_bar(self):
        rng = np.random.default_rng(0)
        tables = accessible_tables(1, 8, 2, 300)
        for table, final_mask in zip(tables, rng.integers(1 << 8, size=300)):
            exact = _class_preserving_fraction(table, int(final_mask), 40, -1.0)
            early = _class_preserving_fraction(table, int(final_mask), 40, 0.02)
            if early < 0:
                self.assertLess(exact, 0.02)
            else:
                self.assertAlmostEqual(early, exact, delta=1e-12)


class TestSampleMinimalAdmittedDFA(unittest.TestCase):
    @parameterized.expand([(4,), (8,)])
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
