"""``min_signal_strength`` is a bound on the oracle's signal, not a description of
it, so an oracle carrying more signal than the caller promised must still be
learned.

The screen used to take its noise rate from the promise: at a declared 0.2 it kept
any suffix disagreeing with the seed on up to ``2 * 0.3 * 0.7 = 0.42`` of the
prefixes.  ``sum mod 9 in {3, 6}`` has non-preserving suffixes at 4/9, close enough
to pass, and a family carrying them cannot separate residues -- ``{0, 3, 6} mod 9``
merged into one accepting state and the learner returned ``sum = 0 (mod 3)``.
"""

import unittest

from orthogonal_dfa.l_star.examples.bernoulli_parity import BernoulliParityOracle
from orthogonal_dfa.l_star.learn import learn_dfa
from orthogonal_dfa.l_star.structures import AsymmetricBernoulli, NoisyOracle
from tests.lstar_common import (
    DEFAULT_SAMPLER,
    assertion_allowed_error,
    evaluate_accuracy,
)

UNDERSTATED_SIGNAL = 0.2


def _oracle_creator(noise_model, seed):
    inner = BernoulliParityOracle(modulo=9, allowed_moduluses=(3, 6))
    return NoisyOracle(inner, noise_model, seed)


class TestUnderstatedSignal(unittest.TestCase):
    def test_a_noiseless_oracle_declared_noisy_is_still_learned(self):
        dfa = learn_dfa(
            _oracle_creator,
            min_signal_strength=UNDERSTATED_SIGNAL,
            seed=0,
            noise_model=AsymmetricBernoulli(p_0=0.0, p_1=1.0),
        )
        accuracy = evaluate_accuracy(dfa, _oracle_creator, sampler=DEFAULT_SAMPLER)
        self.assertGreaterEqual(accuracy, 1 - assertion_allowed_error)
        self.assertEqual(len(dfa.states), 9)


if __name__ == "__main__":
    unittest.main()
