"""Bands not centred on a half (#319), a test per seed: which seeds fail is a
property of the draws, so a bar on one seed says nothing about the rate."""

import unittest

from parameterized import parameterized

from orthogonal_dfa.l_star.examples.bernoulli_parity import BernoulliParityOracle
from orthogonal_dfa.l_star.learn import learn_dfa
from orthogonal_dfa.l_star.structures import AsymmetricBernoulli, NoisyOracle
from tests.e2e.test_merge_traps import assert_armed_not_merged
from tests.lstar_common import DEFAULT_SAMPLER, assertDFA

#: Straddling a half off-centre, at the armed target's signal of 0.3 and above.
ARMED_BANDS = [(0.05, 0.65), (0.15, 0.75), (0.25, 0.95)]
ARMED_SEEDS = 3

#: Both rates below a half.  Seed 0 is in test_asymmetric; above a half, two of
#: four seeds ran past ten minutes.
MODULO_BANDS = [(0.10, 0.40)]
MODULO_SEEDS = range(1, 3)


class TestArmedOffCentre(unittest.TestCase):
    @parameterized.expand(
        [(p_0, p_1, seed) for p_0, p_1 in ARMED_BANDS for seed in range(ARMED_SEEDS)]
    )
    def test_the_armed_state_is_not_merged(self, p_0, p_1, seed):
        assert_armed_not_merged(
            self,
            seed=seed,
            min_signal_strength=(p_1 - p_0) / 2,
            noise_model=AsymmetricBernoulli(p_0=p_0, p_1=p_1),
        )


class TestModuloNonStraddling(unittest.TestCase):
    @parameterized.expand(
        [(p_0, p_1, seed) for p_0, p_1 in MODULO_BANDS for seed in MODULO_SEEDS]
    )
    def test_learned(self, p_0, p_1, seed):
        oracle_creator = lambda nm, s: NoisyOracle(
            BernoulliParityOracle(modulo=9, allowed_moduluses=(3, 6)), nm, s
        )
        dfa = learn_dfa(
            oracle_creator,
            min_signal_strength=(p_1 - p_0) / 2,
            seed=seed,
            noise_model=AsymmetricBernoulli(p_0=p_0, p_1=p_1),
        )
        assertDFA(self, dfa, oracle_creator, sampler=DEFAULT_SAMPLER)


if __name__ == "__main__":
    unittest.main()
