"""``build_pst``/``learn_dfa`` let a caller supply the sampler that draws probe
strings, so a learner can work over a string distribution other than uniform."""

import unittest

import numpy as np

from orthogonal_dfa.l_star.examples.bernoulli_parity import BernoulliParityOracle
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.sampler import Sampler, UniformSampler


class _ZeroRunSampler(Sampler):
    """Emits ``0`` everywhere: a distribution nothing uniform would produce, so a
    tracker built from it is recognisable."""

    def __init__(self, length: int):
        self.length = length

    def sample(self, rng, alphabet_size):
        assert alphabet_size >= 1
        return [0] * self.length


def _oracle(noise_model, seed):
    return BernoulliParityOracle(noise_model, seed)


class TestCustomSampler(unittest.TestCase):
    def test_defaults_to_uniform_over_sample_length(self):
        pst = build_pst(_oracle, min_signal_strength=0.3, seed=0, sample_length=15)
        self.assertIsInstance(pst.sampler, UniformSampler)
        self.assertEqual(pst.sampler.length, 15)

    def test_custom_sampler_is_used(self):
        sampler = _ZeroRunSampler(12)
        pst = build_pst(_oracle, min_signal_strength=0.3, seed=0, sampler=sampler)
        self.assertIs(pst.sampler, sampler)
        # The prefixes come from the sampler, so they carry its shape rather than
        # the uniform default's.
        sampled = [p for p in pst.table.prefixes if len(p) == sampler.length]
        self.assertTrue(sampled, "no prefix of the sampler's length was drawn")
        for prefix in sampled:
            self.assertEqual(prefix, [0] * sampler.length)

    def test_sample_length_ignored_when_sampler_given(self):
        sampler = _ZeroRunSampler(7)
        pst = build_pst(
            _oracle,
            min_signal_strength=0.3,
            seed=0,
            sample_length=40,
            sampler=sampler,
        )
        self.assertEqual(pst.sampler.length, 7)

    def test_sampler_length_is_part_of_the_interface(self):
        # The learner reads `length` off whatever sampler it is given.
        self.assertEqual(UniformSampler(9).length, 9)
        self.assertEqual(_ZeroRunSampler(9).length, 9)
        self.assertEqual(len(UniformSampler(9).sample(np.random.default_rng(0), 2)), 9)


if __name__ == "__main__":
    unittest.main()
