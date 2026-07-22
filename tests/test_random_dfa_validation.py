"""
Makes sure that the preconditions for learnability admit only DFAs that E-L* can learn.
"""

import signal
import unittest

import numpy as np

from orthogonal_dfa.l_star import preconditions as P
from orthogonal_dfa.l_star.examples.benchmark_generator import (
    DFAOracle,
    sample_random_dfa,
)
from tests.test_lstar import compute_dfa_accuracy, compute_dfa_for_oracle

NUM_DFAS = 600
ETA = 0.05
LENGTH = 40
LEARNED = 0.95  # E-L* accuracy bar for "learned it"
PER_DFA_TIMEOUT = 60  # a passer that hangs this long is itself a failure


def _elstar_accuracy(aut) -> float:
    """Noiseless accuracy of the DFA E-L* learns for ``aut``; 0.0 if it gave
    up, errored, or exceeded ``PER_DFA_TIMEOUT``."""

    def oracle_creator(name, seed):
        return DFAOracle(name, seed, aut)

    def _timeout(*_):
        raise TimeoutError

    old = signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(PER_DFA_TIMEOUT)
    try:
        _, dfa, _ = compute_dfa_for_oracle(
            oracle_creator, min_signal_strength=0.5 - ETA, seed=0
        )
        if dfa is None:
            return 0.0
        acc, _, _ = compute_dfa_accuracy(dfa, oracle_creator)
        return acc
    except Exception:  # pylint: disable=broad-exception-caught
        return 0.0  # gave up, timed out, or errored -- all "did not learn"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


class TestPreconditionsNoFalsePositives(unittest.TestCase):
    def test_admitted_random_dfas_are_learnable(self):
        rng = np.random.default_rng(0)
        false_positives = []
        for i in range(NUM_DFAS):
            n = int(rng.integers(2, 6))
            aut = sample_random_dfa(rng, num_states=n, alphabet_size=2)
            if not P.satisfies_preconditions(aut, length=LENGTH):
                continue
            acc = _elstar_accuracy(aut)
            if acc < LEARNED:
                false_positives.append((i, n, round(acc, 3)))
        self.assertEqual(
            false_positives,
            [],
            f"preconditions admitted DFAs E-L* could not learn "
            f"(index, states, accuracy): {false_positives}",
        )


if __name__ == "__main__":
    unittest.main()
