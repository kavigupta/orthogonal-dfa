"""E-L* on a few targets from `sample_minimal_admitted_dfa`."""

import functools
import unittest

import numpy as np
from parameterized import parameterized

from orthogonal_dfa.l_star.examples.benchmark_generator import DFAOracle
from orthogonal_dfa.l_star.examples.random_dfa import sample_minimal_admitted_dfa
from orthogonal_dfa.l_star.structures import NoisyOracle
from tests.lstar_common import (
    DEFAULT_SAMPLER,
    assertion_allowed_error,
    compute_dfa_accuracy,
)
from tests.lstar_common import learn_dfa_verified as learn_dfa

NUM_DFAS = 5
NUM_STATES = 8
ETA = 0.05
LENGTH = 40


@functools.lru_cache(maxsize=None)
def _targets():
    # drawn on first use rather than at import, since every e2e shard collects
    # this file
    rng = np.random.default_rng(0)
    return [
        sample_minimal_admitted_dfa(
            rng, num_states=NUM_STATES, alphabet_size=2, length=LENGTH
        )
        for _ in range(NUM_DFAS)
    ]


class TestRandomMinimalDFAs(unittest.TestCase):
    @parameterized.expand([(i,) for i in range(NUM_DFAS)])
    def test_learned(self, index):
        target = _targets()[index]
        oracle_creator = lambda nm, s, _dfa=target: NoisyOracle(DFAOracle(_dfa), nm, s)
        dfa = learn_dfa(oracle_creator, min_signal_strength=0.5 - ETA, seed=0)
        accuracy, fp, fn = compute_dfa_accuracy(
            dfa, oracle_creator, sampler=DEFAULT_SAMPLER
        )
        self.assertGreaterEqual(
            accuracy,
            1 - assertion_allowed_error,
            f"accuracy {accuracy:.3f}, FP: {len(fp)}, FN: {len(fn)}",
        )


if __name__ == "__main__":
    unittest.main()
