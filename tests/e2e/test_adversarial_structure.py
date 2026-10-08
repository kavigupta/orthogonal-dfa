"""Targets whose states are reached or told apart only through long, specific strings."""

import unittest

from automata.fa.dfa import DFA
from parameterized import parameterized

from orthogonal_dfa.l_star import preconditions as P
from orthogonal_dfa.l_star.examples.benchmark_generator import DFAOracle
from orthogonal_dfa.l_star.learn import learn_dfa as learn_dfa_unchecked
from orthogonal_dfa.l_star.structures import NoisyOracle
from tests.lstar_common import DEFAULT_SAMPLER, assertDFA
from tests.lstar_common import learn_dfa_verified as learn_dfa


class TestLStarDeepCounter(unittest.TestCase):
    """``Sigma* 0^k Sigma*`` — "contains a run of k zeros".

    State i counts i consecutive zeros, so its only access string is ``0^i``, and
    the deepest counter states are reached only via long, specific paths that
    random length-L probes almost never hit. They are nonetheless recurrent and
    on the critical path to acceptance. This guards that counterexample-driven
    discovery still finds and enriches them — a misclassified ``0^k`` string
    yields a discriminating prefix ending exactly at a deep state, which seeds it.
    """

    @parameterized.expand([(k,) for k in (6, 7)])
    def test_contains_run_of_k_zeros(self, k):
        transitions = {i: {0: i + 1, 1: 0} for i in range(k)}
        transitions[k] = {0: k, 1: k}  # absorbing accept once k zeros are seen
        dfa = DFA(
            states=set(range(k + 1)),
            input_symbols={0, 1},
            transitions=transitions,
            initial_state=0,
            final_states={k},
            allow_partial=False,
        )
        oracle_creator = lambda nm, s, _d=dfa: NoisyOracle(DFAOracle(_d), nm, s)
        learned = learn_dfa(oracle_creator, min_signal_strength=0.3, seed=0)
        assertDFA(self, learned, oracle_creator, sampler=DEFAULT_SAMPLER)


# A mod-4 counter crossed with a pattern: the consistency gate passes a
# hypothesis that is the counter alone, at an accuracy near 0.93.

COUNTED_PATTERN = (0, 1, 1, 0, 0)
COUNTED_MODULUS = 4
COUNTED_RESIDUE = 2


def build_counted_pattern() -> DFA:
    """Accepts strings whose count of 1s is COUNTED_RESIDUE mod
    COUNTED_MODULUS and that contain COUNTED_PATTERN."""

    def matched(done, symbol):
        """Longest prefix of the pattern that ends the string read so far."""
        if done == len(COUNTED_PATTERN):
            return done
        text = COUNTED_PATTERN[:done] + (symbol,)
        for length in range(len(text), -1, -1):
            if text[len(text) - length :] == COUNTED_PATTERN[:length]:
                return length
        return 0

    states = [
        (count, done)
        for count in range(COUNTED_MODULUS)
        for done in range(len(COUNTED_PATTERN) + 1)
    ]
    return DFA(
        states=set(states),
        input_symbols={0, 1},
        transitions={
            (count, done): {
                c: ((count + c) % COUNTED_MODULUS, matched(done, c)) for c in (0, 1)
            }
            for count, done in states
        },
        initial_state=(0, 0),
        final_states={(COUNTED_RESIDUE, len(COUNTED_PATTERN))},
        allow_partial=False,
    ).minify()


class TestCountedPattern(unittest.TestCase):
    def test_admitted(self):
        report = P.satisfies_preconditions(
            build_counted_pattern(), length=DEFAULT_SAMPLER.length, short_circuit=False
        )
        self.assertTrue(report.satisfied, report.reasons)

    @parameterized.expand([(seed,) for seed in range(3)])
    def test_the_pattern_is_learned(self, seed):
        target = build_counted_pattern()
        oracle_creator = lambda nm, s, _d=target: NoisyOracle(DFAOracle(_d), nm, s)
        dfa = learn_dfa_unchecked(oracle_creator, min_signal_strength=0.3, seed=seed)
        assertDFA(self, dfa, oracle_creator, sampler=DEFAULT_SAMPLER)
