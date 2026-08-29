"""Noise as something applied to an oracle rather than owned by one."""

import unittest

import numpy as np

from orthogonal_dfa.l_star.examples.bernoulli_parity import BernoulliParityOracle
from orthogonal_dfa.l_star.structures import NoisyOracle, Oracle, SymmetricBernoulli

CLEAN = SymmetricBernoulli(p_correct=1.0)
NOISY = SymmetricBernoulli(p_correct=0.7)


class _Parity(Oracle):
    """Even parity, and nothing about noise."""

    alphabet_size = 2

    def membership_query(self, string):
        return sum(string) % 2 == 0

    def target_dfa(self):
        return NoisyOracle(BernoulliParityOracle(), CLEAN, 0).target_dfa()


def _strings(count=400, length=8, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 2, size=length).tolist() for _ in range(count)]


class TestNoisyOracle(unittest.TestCase):
    def test_a_clean_noise_model_changes_nothing(self):
        inner = _Parity()
        wrapped = NoisyOracle(inner, CLEAN, 0)
        strings = _strings()
        np.testing.assert_array_equal(
            wrapped.membership_queries(strings),
            [inner.membership_query(s) for s in strings],
        )

    def test_it_flips_about_the_rate_it_is_given(self):
        inner = _Parity()
        wrapped = NoisyOracle(inner, NOISY, 0)
        strings = _strings(4000)
        agree = np.mean(
            [
                w == inner.membership_query(s)
                for w, s in zip(wrapped.membership_queries(strings), strings)
            ]
        )
        self.assertAlmostEqual(agree, 0.7, delta=4 * np.sqrt(0.7 * 0.3 / 4000))

    def test_the_same_string_gets_the_same_answer(self):
        wrapped = NoisyOracle(_Parity(), NOISY, 0)
        strings = _strings(200)
        np.testing.assert_array_equal(
            wrapped.membership_queries(strings), wrapped.membership_queries(strings)
        )

    def test_a_batch_answers_as_the_queries_do(self):
        wrapped = NoisyOracle(_Parity(), NOISY, 0)
        strings = _strings(200)
        np.testing.assert_array_equal(
            wrapped.membership_queries(strings),
            [wrapped.membership_query(s) for s in strings],
        )

    def test_the_seed_moves_which_answers_flip(self):
        strings = _strings(400)
        first = NoisyOracle(_Parity(), NOISY, 0).membership_queries(strings)
        second = NoisyOracle(_Parity(), NOISY, 1).membership_queries(strings)
        self.assertFalse(np.array_equal(first, second))

    def test_the_language_is_what_it_was(self):
        # Noise moves answers, not the language they are answers about, so a
        # caller reasoning about the target still gets the inner one.
        inner = _Parity()
        self.assertEqual(NoisyOracle(inner, NOISY, 0).target_dfa(), inner.target_dfa())

    def test_the_alphabet_is_what_it_was(self):
        self.assertEqual(NoisyOracle(_Parity(), NOISY, 0).alphabet_size, 2)


if __name__ == "__main__":
    unittest.main()
