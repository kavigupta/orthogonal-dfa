"""Targets where a hypothesis's transitions agree with the target at one position or parity.

A learner that checks the hypothesis's transitions only at one position (or one parity) accepts
a wrong DFA on these; a run from the start over the whole string exposes it.
"""

import unittest

from automata.fa.dfa import DFA
from parameterized import parameterized

from orthogonal_dfa.l_star import preconditions as P
from orthogonal_dfa.l_star.examples.benchmark_generator import DFAOracle
from orthogonal_dfa.l_star.learn import learn_dfa as learn_dfa_unchecked
from orthogonal_dfa.l_star.structures import NoisyOracle
from tests.lstar_common import DEFAULT_SAMPLER, assertDFA

RARE_TOGGLE_PERIOD = 4


def build_parity_toggle() -> DFA:
    """States (length parity, bit), accepting iff bit. At odd positions 0 toggles the bit."""
    states = [(par, bit) for par in (0, 1) for bit in (0, 1)]

    def step(state, c):
        par, bit = state
        if par == 1 and c == 0:
            bit = 1 - bit
        return (1 - par, bit)

    return DFA(
        states=set(states),
        input_symbols={0, 1},
        transitions={s: {c: step(s, c) for c in (0, 1)} for s in states},
        initial_state=(0, 0),
        final_states={s for s in states if s[1] == 1},
        allow_partial=False,
    )


def build_rare_toggle(m: int) -> DFA:
    """States (bit, j), j = trailing 0s mod m, accepting iff bit. The m-th 0 in a row toggles the bit."""
    states = [(bit, j) for bit in (0, 1) for j in range(m)]

    def step(state, c):
        bit, j = state
        if c == 1:
            return (bit, 0)
        if j == m - 1:
            return (1 - bit, 0)
        return (bit, j + 1)

    return DFA(
        states=set(states),
        input_symbols={0, 1},
        transitions={s: {c: step(s, c) for c in (0, 1)} for s in states},
        initial_state=(0, 0),
        final_states={s for s in states if s[0] == 1},
        allow_partial=False,
    )


class TestWalkParity(unittest.TestCase):
    def check_learned(self, target, seed):
        oracle_creator = lambda nm, s, _d=target: NoisyOracle(DFAOracle(_d), nm, s)
        dfa = learn_dfa_unchecked(oracle_creator, min_signal_strength=0.3, seed=seed)
        assertDFA(self, dfa, oracle_creator, sampler=DEFAULT_SAMPLER)

    def test_parity_toggle_admitted(self):
        report = P.satisfies_preconditions(
            build_parity_toggle(), length=DEFAULT_SAMPLER.length, short_circuit=False
        )
        self.assertTrue(report.satisfied, report.reasons)

    def test_rare_toggle_admitted(self):
        report = P.satisfies_preconditions(
            build_rare_toggle(RARE_TOGGLE_PERIOD),
            length=DEFAULT_SAMPLER.length,
            short_circuit=False,
        )
        self.assertTrue(report.satisfied, report.reasons)

    @parameterized.expand([(seed,) for seed in range(3)])
    def test_parity_toggle_is_learned(self, seed):
        self.check_learned(build_parity_toggle(), seed)

    @parameterized.expand([(seed,) for seed in range(3)])
    def test_rare_toggle_is_learned(self, seed):
        self.check_learned(build_rare_toggle(RARE_TOGGLE_PERIOD), seed)
