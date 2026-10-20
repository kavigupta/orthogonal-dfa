"""Runs whose probes walk deep from round 0, so that the start check's
resumption and the shallow populations' drop are exercised."""

import unittest

from orthogonal_dfa.l_star.examples.benchmark_generator import (
    DFAOracle,
    sample_balanced_benchmark,
)
from orthogonal_dfa.l_star.preconditions import start_length
from orthogonal_dfa.l_star.structures import NoisyOracle
from tests.e2e.test_basic import ANOTHER_POOR_CASE_TARGET
from tests.e2e.test_walk_parity import build_parity_toggle
from tests.lstar_common import DEFAULT_SAMPLER, assert_terminates, assertDFA
from tests.lstar_common import learn_dfa_verified as learn_dfa

DEEPEST = start_length(DEFAULT_SAMPLER.length)


class TestWalkedFromHalfway(unittest.TestCase):
    def test_generated_benchmark_7(self):
        outer, _, _ = sample_balanced_benchmark(
            7,
            alphabet_size=2,
            num_inner_states=12,
            num_outer_states=10,
            probe_length=40,
        )
        oracle_creator = lambda nm, s, _dfa=outer: NoisyOracle(DFAOracle(_dfa), nm, s)
        dfa = learn_dfa(
            oracle_creator, min_signal_strength=0.3, seed=0, initial_k=DEEPEST
        )
        assertDFA(self, dfa, oracle_creator, sampler=DEFAULT_SAMPLER)

    def test_poor_case_terminates_at_the_default_bar(self):
        target = ANOTHER_POOR_CASE_TARGET
        oracle_creator = lambda nm, s, _dfa=target: NoisyOracle(DFAOracle(_dfa), nm, s)
        assert_terminates(
            lambda: learn_dfa(
                oracle_creator, min_signal_strength=0.3, seed=0, initial_k=DEEPEST
            ),
            seconds=300,
            message="synthesis did not terminate within the timeout",
        )


class TestWalkedFromAQuarter(unittest.TestCase):
    def test_parity_toggle_is_learned(self):
        target = build_parity_toggle()
        oracle_creator = lambda nm, s, _d=target: NoisyOracle(DFAOracle(_d), nm, s)
        dfa = learn_dfa(
            oracle_creator, min_signal_strength=0.3, seed=1, initial_k=DEEPEST // 2
        )
        assertDFA(self, dfa, oracle_creator, sampler=DEFAULT_SAMPLER)


if __name__ == "__main__":
    unittest.main()
