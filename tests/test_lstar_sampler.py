"""``build_pst``/``learn_dfa`` let a caller supply the sampler that draws probe
strings, so a learner can work over a string distribution other than uniform."""

import unittest

import numpy as np

from orthogonal_dfa.l_star.examples.bernoulli_parity import BernoulliParityOracle
from orthogonal_dfa.l_star.learn import DEFAULT_SAMPLE_LENGTH, build_pst
from orthogonal_dfa.l_star.sampler import Sampler, UniformSampler
from orthogonal_dfa.l_star.structures import NoisyOracle


class _ZeroRunSampler(Sampler):
    """Emits ``0`` everywhere: a distribution nothing uniform would produce, so a
    tracker built from it is recognisable."""

    def __init__(self, length: int):
        self.length = length

    def sample(self, rng, alphabet_size):
        assert alphabet_size >= 1
        return [0] * self.length

    def symbol_weights(self, alphabet_size):
        return [1] + [0] * (alphabet_size - 1)


def _oracle(noise_model, seed):
    return NoisyOracle(BernoulliParityOracle(), noise_model, seed)


class TestCustomSampler(unittest.TestCase):
    def test_defaults_to_uniform(self):
        pst = build_pst(_oracle, min_signal_strength=0.3, seed=0)
        self.assertIsInstance(pst.sampler, UniformSampler)
        self.assertEqual(pst.sampler.length, DEFAULT_SAMPLE_LENGTH)

    def test_length_is_varied_through_the_sampler(self):
        # There is one knob, not two: a different probe length is a different
        # UniformSampler.
        pst = build_pst(
            _oracle, min_signal_strength=0.3, seed=0, sampler=UniformSampler(15)
        )
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

    def test_sampler_length_is_part_of_the_interface(self):
        # The learner reads `length` off whatever sampler it is given.
        self.assertEqual(UniformSampler(9).length, 9)
        self.assertEqual(_ZeroRunSampler(9).length, 9)
        self.assertEqual(len(UniformSampler(9).sample(np.random.default_rng(0), 2)), 9)


if __name__ == "__main__":
    unittest.main()
