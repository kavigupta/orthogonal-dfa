"""Noise models whose two classes flip at different rates."""

import unittest

from parameterized import parameterized

from orthogonal_dfa.l_star.examples.bernoulli_parity import (
    BernoulliParityOracle,
    BernoulliRegex,
)
from orthogonal_dfa.l_star.structures import AsymmetricBernoulli, NoisyOracle
from tests.lstar_common import DEFAULT_SAMPLER, assert_modulo_skewed_learned, assertDFA
from tests.lstar_common import learn_dfa_verified as learn_dfa


class TestLStarAsymmetric(unittest.TestCase):
    def test_regex_asymmetric(self):
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            BernoulliRegex(regex=r".*1010101.*"), noise_model, seed
        )
        noise_model = AsymmetricBernoulli(p_0=0.15, p_1=0.7)
        # signal = (0.7 - 0.15) / 2 = 0.275, but for now we're using 0.2 to be safe.
        dfa = learn_dfa(
            oracle_creator, min_signal_strength=0.2, seed=0, noise_model=noise_model
        )
        assertDFA(self, dfa, oracle_creator, sampler=DEFAULT_SAMPLER)

    @parameterized.expand([(0.10, 0.40), (0.60, 0.90)])
    def test_modulo_asymmetric_non_straddling(self, p_0, p_1):
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            BernoulliParityOracle(modulo=9, allowed_moduluses=(3, 6)), noise_model, seed
        )
        noise_model = AsymmetricBernoulli(p_0=p_0, p_1=p_1)
        # signal = (p_1 - p_0) / 2, so 0.15 in both cases.
        dfa = learn_dfa(
            oracle_creator, min_signal_strength=0.15, seed=0, noise_model=noise_model
        )
        assertDFA(self, dfa, oracle_creator, sampler=DEFAULT_SAMPLER)

    # Seed 0 is in TestLStarAsymmetricFast.
    @parameterized.expand([(seed,) for seed in range(1, 12)])
    def test_modulo_skewed_understated(self, seed):
        assert_modulo_skewed_learned(self, seed=seed)

    def test_one_sided_noise(self):
        """One class is pure coin-flip (p_0=0.50), only the other carries signal."""
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            BernoulliParityOracle(modulo=9, allowed_moduluses=(3, 6)), noise_model, seed
        )
        noise_model = AsymmetricBernoulli(p_0=0.50, p_1=0.80)
        # signal = 0.15, boundary = 0.65
        dfa = learn_dfa(
            oracle_creator, min_signal_strength=0.15, seed=0, noise_model=noise_model
        )
        assertDFA(self, dfa, oracle_creator, sampler=DEFAULT_SAMPLER)


class TestLStarAsymmetricFast(unittest.TestCase):
    def test_modulo_asymmetric(self):
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            BernoulliParityOracle(modulo=9, allowed_moduluses=(3, 6)), noise_model, seed
        )
        noise_model = AsymmetricBernoulli(p_0=0.05, p_1=0.85)
        # signal = (0.85 - 0.05) / 2 = 0.4, but for now we're using 0.35 to be safe.
        dfa = learn_dfa(
            oracle_creator, min_signal_strength=0.35, seed=0, noise_model=noise_model
        )
        assertDFA(self, dfa, oracle_creator, sampler=DEFAULT_SAMPLER)

    def test_modulo_asymmetric_skewed(self):
        assert_modulo_skewed_learned(self, seed=0)

    def test_rare_accept_class(self):
        """Only 1 of 7 states is accepting, so boundary estimation sees mostly rejects."""
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            BernoulliParityOracle(modulo=7, allowed_moduluses=(3,)), noise_model, seed
        )
        noise_model = AsymmetricBernoulli(p_0=0.15, p_1=0.75)
        # signal = 0.30, boundary = 0.45
        dfa = learn_dfa(
            oracle_creator, min_signal_strength=0.25, seed=0, noise_model=noise_model
        )
        assertDFA(self, dfa, oracle_creator, sampler=DEFAULT_SAMPLER)

    @unittest.skip(
        "Trimodal over seeds -- 4/8 resolve 9 states, 2/8 collapse to 3, 2/8 raise "
        "from synthesis -- so no single-seed assertion holds either way. See #230."
    )
    def test_boundary_near_zero(self):
        """Both noise rates near 0, boundary far from 0.5.
        The mode this pins: finds only 3 states instead of 9. With the true
        boundary at 0.22, the clustering threshold is so low that true-reject
        prefixes (mean ~0.02) get mixed into the "accept" group on noisy suffix
        samples, contaminating the boundary estimate downward to ~0.11."""
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            BernoulliParityOracle(modulo=9, allowed_moduluses=(3, 6)), noise_model, seed
        )
        noise_model = AsymmetricBernoulli(p_0=0.02, p_1=0.42)
        # signal = 0.20, boundary = 0.22
        dfa = learn_dfa(
            oracle_creator, min_signal_strength=0.15, seed=0, noise_model=noise_model
        )
        assertDFA(self, dfa, oracle_creator, sampler=DEFAULT_SAMPLER)
