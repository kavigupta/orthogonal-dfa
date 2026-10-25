"""
Makes sure that the preconditions for learnability admit only DFAs that E-L* can learn.
"""

import signal
import unittest

import numpy as np
from parameterized import parameterized

from orthogonal_dfa.l_star.examples.benchmark_generator import DFAOracle
from orthogonal_dfa.l_star.examples.random_dfa import sample_admitted_dfas
from orthogonal_dfa.l_star.structures import NoisyOracle
from tests.lstar_common import DEFAULT_SAMPLER, compute_dfa_accuracy
from tests.lstar_common import learn_dfa_verified as learn_dfa

NUM_DFAS = 37
ETA = 0.05
LENGTH = 40
LEARNED = 0.95  # E-L* accuracy bar for "learned it"
#: Only admitted DFAs are run, and the slowest measured is ~113s, so this is
#: a backstop against a hang rather than a learnability standard.
PER_DFA_TIMEOUT = 600


ADMITTED_DFAS = [
    (f"{i}_states_{len(aut.states)}", aut)
    for i, aut in enumerate(
        sample_admitted_dfas(
            np.random.default_rng(0),
            count=NUM_DFAS,
            min_states=2,
            max_states=5,
            alphabet_size=2,
            length=LENGTH,
        )
    )
]


def _elstar_accuracy(aut) -> float:
    """Noiseless accuracy of the DFA E-L* learns for ``aut``; 0.0 if it gave up.

    Only the learning is under the alarm, so a hang raises ``TimeoutError``
    rather than being scored as a DFA the preconditions wrongly admitted.
    """

    def oracle_creator(name, seed):
        return NoisyOracle(DFAOracle(aut), name, seed)

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
    acc, _, _ = compute_dfa_accuracy(dfa, oracle_creator, sampler=DEFAULT_SAMPLER)
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
