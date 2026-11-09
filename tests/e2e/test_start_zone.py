"""E-L* on targets the start-zone precondition rejects: the decision rides on
states the window never or rarely lands in."""

import functools
import unittest

import numpy as np
from automata.fa.dfa import DFA
from parameterized import parameterized

from orthogonal_dfa.l_star.examples.benchmark_generator import DFAOracle
from orthogonal_dfa.l_star.examples.bernoulli_parity import BernoulliRegex
from orthogonal_dfa.l_star.examples.random_dfa import sample_minimal_admitted_dfa
from orthogonal_dfa.l_star.structures import NoisyOracle
from tests.lstar_common import DEFAULT_SAMPLER, assertDFA
from tests.lstar_common import learn_dfa_verified as learn_dfa


def _injective_head():
    """The first three letters pick one of eight head states, each starting a
    mod-9 count of ones at its own phase."""
    heads = [format(i, "03b") for i in range(8)]
    phases = [(a, b) for a in range(9) for b in range(9) if b != (a + 1) % 9][::10]
    transitions = {}
    for depth in range(3):
        for i in range(2**depth):
            v = format(i, f"0{depth}b") if depth else ""
            transitions["h" + v] = {0: "h" + v + "0", 1: "h" + v + "1"}
    for u, (a, b) in zip(heads, phases):
        transitions["h" + u] = {0: f"r{a}", 1: f"r{b}"}
    for r in range(9):
        transitions[f"r{r}"] = {0: f"r{r}", 1: f"r{(r + 1) % 9}"}
    return DFA(
        states=set(transitions),
        input_symbols={0, 1},
        transitions=transitions,
        initial_state="h",
        final_states={"r3", "r6"},
    )


@functools.lru_cache(maxsize=None)
def _random_target(seed):
    return sample_minimal_admitted_dfa(
        np.random.default_rng((18, seed)), num_states=18, alphabet_size=2, length=40
    )


class TestStartZone(unittest.TestCase):
    @parameterized.expand(
        [
            ("four_ones", r"1111.*"),
            ("zero_one_one_zero", r"0110.*"),
            # CAPAL's Simple04 and Difficult10.
            ("first_letter", r"0.*"),
            ("first_letter_sets_parity", r"0([01][01])*|1[01]([01][01])*"),
        ]
    )
    def test_regex(self, _, regex):
        oracle_creator = lambda nm, s: NoisyOracle(BernoulliRegex(regex=regex), nm, s)
        dfa = learn_dfa(oracle_creator, min_signal_strength=0.25, seed=0)
        assertDFA(self, dfa, oracle_creator, sampler=DEFAULT_SAMPLER)

    def test_injective_head(self):
        target = _injective_head()
        oracle_creator = lambda nm, s: NoisyOracle(DFAOracle(target), nm, s)
        dfa = learn_dfa(oracle_creator, min_signal_strength=0.45, seed=0)
        assertDFA(self, dfa, oracle_creator, sampler=DEFAULT_SAMPLER)

    # 1 drains a 16-state transient head into two sinks; main's walk start fails
    # 31 and 35.
    @parameterized.expand([(1,), (31,), (35,)])
    def test_random_18_states(self, seed):
        target = _random_target(seed)
        oracle_creator = lambda nm, s: NoisyOracle(DFAOracle(target), nm, s)
        dfa = learn_dfa(oracle_creator, min_signal_strength=0.45, seed=0)
        assertDFA(self, dfa, oracle_creator, sampler=DEFAULT_SAMPLER)


if __name__ == "__main__":
    unittest.main()
