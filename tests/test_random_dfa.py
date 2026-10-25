import unittest

import numpy as np
from parameterized import parameterized

from orthogonal_dfa.l_star import preconditions as P
from orthogonal_dfa.l_star.examples.random_dfa import (
    _functional_graph_components,
    sample_minimal_admitted_dfa,
)


class TestFunctionalGraphComponents(unittest.TestCase):
    def test_components(self):
        # 0 <-> 1 <- 2;  3 <-> 4 <- 5;  6 -> 6
        labels = _functional_graph_components(np.array([1, 0, 1, 4, 3, 3, 6]))
        self.assertEqual(labels.tolist(), [0, 0, 0, 3, 3, 3, 6])

    def test_chain_into_a_cycle(self):
        labels = _functional_graph_components(np.array([1, 2, 3, 4, 5, 5]))
        self.assertEqual(set(labels.tolist()), {0})


class TestSampleMinimalAdmittedDFA(unittest.TestCase):
    @parameterized.expand([(4,), (8,)])
    def test_minimal_of_the_size_and_admitted(self, num_states):
        for i in range(5):
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
