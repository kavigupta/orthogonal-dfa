"""
Makes sure that the preconditions for learnability admit only DFAs that E-L* can learn.
"""

import signal
import unittest

import numpy as np
from parameterized import parameterized

from orthogonal_dfa.l_star import preconditions as P
from orthogonal_dfa.l_star.examples.benchmark_generator import (
    DFAOracle,
    sample_random_dfa,
)
from tests.lstar_common import compute_dfa_accuracy
from tests.lstar_common import learn_dfa_verified as learn_dfa

NUM_DFAS = 600
ETA = 0.05
LENGTH = 40
LEARNED = 0.95  # E-L* accuracy bar for "learned it"
#: Only admitted DFAs are run, and the slowest measured is ~113s, so this is
#: a backstop against a hang rather than a learnability standard.
PER_DFA_TIMEOUT = 600


def _admitted_dfas():
    """``(name, dfa)`` for each sampled DFA the preconditions admit."""
    rng = np.random.default_rng(0)
    admitted = []
    for i in range(NUM_DFAS):
        num_states = int(rng.integers(2, 6))
        aut = sample_random_dfa(rng, num_states=num_states, alphabet_size=2)
        if P.satisfies_preconditions(aut, length=LENGTH):
            admitted.append((f"{i}_states_{num_states}", aut))
    return admitted


ADMITTED_DFAS = _admitted_dfas()


def _elstar_accuracy(aut) -> float:
    """Noiseless accuracy of the DFA E-L* learns for ``aut``; 0.0 if it gave up.

    Only the learning is under the alarm, so a hang raises ``TimeoutError``
    rather than being scored as a DFA the preconditions wrongly admitted.
    """

    def oracle_creator(name, seed):
        return DFAOracle(name, seed, aut)

    def _timeout(*_):
        raise TimeoutError(f"E-L* did not terminate within {PER_DFA_TIMEOUT}s")

    old = signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(PER_DFA_TIMEOUT)
    try:
        dfa = learn_dfa(oracle_creator, min_signal_strength=0.5 - ETA, seed=0)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    if dfa is None:
        return 0.0
    acc, _, _ = compute_dfa_accuracy(dfa, oracle_creator)
    return acc


class TestPreconditionsNoFalsePositives(unittest.TestCase):
    @parameterized.expand(ADMITTED_DFAS)
    def test_admitted_random_dfa_is_learnable(self, _name, aut):
        acc = _elstar_accuracy(aut)
        self.assertGreaterEqual(
            acc,
            LEARNED,
            f"preconditions admitted a DFA E-L* could not learn (accuracy {acc:.3f})",
        )


if __name__ == "__main__":
    unittest.main()
