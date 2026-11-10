"""Random-minimal experiment cells that failed on #676, each run exactly as the
experiment runs it."""

import random
import unittest

import numpy as np
import pytest
from parameterized import parameterized

from orthogonal_dfa.experiments.capal_comparison.core import run_elstar_cell
from orthogonal_dfa.l_star.examples.benchmark_generator import DFAOracle
from orthogonal_dfa.l_star.examples.random_dfa import dfa_from_table
from orthogonal_dfa.l_star.structures import NoisyOracle

# name -> (transition table, final-state bitmask), as drawn into
# data/random_minimal/targets on #676.
TARGETS = {
    "n12_098": (
        [[2, 1], [11, 5], [4, 2], [11, 10], [7, 9], [8, 4]]
        + [[6, 6], [7, 7], [5, 7], [10, 10], [6, 0], [3, 7]],
        2239,
    ),
    "n18_064": (
        [[16, 14], [8, 10], [11, 15], [8, 6], [4, 4], [8, 16], [5, 3], [6, 17]]
        + [[3, 2], [15, 17], [9, 12], [13, 2], [7, 2], [1, 17], [15, 7], [7, 4]]
        + [[5, 17], [7, 12]],
        229358,
    ),
    "n18_091": (
        [[7, 6], [2, 0], [8, 15], [9, 9], [1, 3], [12, 3], [8, 12], [5, 16]]
        + [[11, 2], [3, 3], [0, 4], [1, 10], [14, 16], [10, 17], [17, 13]]
        + [[12, 11], [13, 10], [0, 11]],
        261631,
    ),
}


def _words(count=5000, length=40):
    rng = random.Random(0)
    return [[rng.randrange(2) for _ in range(length)] for _ in range(count)]


class TestRandomMinimalFailures(unittest.TestCase):
    @parameterized.expand(
        [
            # NoAcceptPreservingFamily: the gate kept blaming a retired population.
            ("n12_098", 0.2),
            # SourceDry: the gate drew its certification sample from a dry source.
            ("n18_064", 0.3),
            ("n18_091", 0.3),
        ]
    )
    @pytest.mark.slow
    def test_cell(self, name, eta):
        table, final_mask = TARGETS[name]
        target = dfa_from_table(np.array(table), final_mask)
        cell = run_elstar_cell(
            lambda noise_model, s: NoisyOracle(DFAOracle(target), noise_model, s),
            benchmark=name,
            family="random_minimal",
            eta=eta,
            seed=0,
            symbols=2,
            words=_words(),
            truth=target.accepts_input,
            sample_length=40,
            target_states=len(target.states),
        )
        self.assertIsNone(cell.error, cell.error)
        self.assertGreaterEqual(cell.accuracy, 0.98)


if __name__ == "__main__":
    unittest.main()
