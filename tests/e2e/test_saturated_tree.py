"""A target whose language turns on a state only the empty prefix reaches."""

import unittest

from automata.fa.dfa import DFA

from orthogonal_dfa.l_star import preconditions as P
from orthogonal_dfa.l_star.examples.benchmark_generator import DFAOracle
from orthogonal_dfa.l_star.structures import NoisyOracle
from tests.lstar_common import DEFAULT_SAMPLER, assertDFA
from tests.lstar_common import learn_dfa_verified as learn_dfa

#: Residues each branch accepts on.  They differ, so the first symbol decides
#: the language; they overlap, so a suffix that leaves the residue alone
#: preserves the classes and an accept-preserving family still exists.
MOD, ACCEPT_AFTER_ZERO, ACCEPT_AFTER_ONE = 5, {1, 3, 4}, {1, 4}


def _target() -> DFA:
    """``w`` is accepted on ``#1s in w[1:] mod 5``, by a rule ``w[0]`` picks.

    State 0 is left on the first symbol and never re-entered, so the only
    string ending there is the empty one.
    """
    transitions = {0: {0: 1, 1: 1 + MOD}}
    for branch in (0, 1):
        for residue in range(MOD):
            state = 1 + branch * MOD + residue
            transitions[state] = {
                0: state,
                1: 1 + branch * MOD + (residue + 1) % MOD,
            }
    accepting = {
        1 + branch * MOD + residue
        for branch, accept in ((0, ACCEPT_AFTER_ZERO), (1, ACCEPT_AFTER_ONE))
        for residue in range(MOD)
        if residue in accept
    }
    return DFA(
        states=set(range(1 + 2 * MOD)),
        input_symbols={0, 1},
        transitions=transitions,
        initial_state=0,
        final_states=accepting,
        allow_partial=False,
    )


class TestSaturatedTree(unittest.TestCase):
    def test_only_the_empty_prefix_reaches_the_deciding_state(self):
        report = P.satisfies_preconditions(_target(), length=40, short_circuit=False)

        self.assertTrue(report, report.reasons)
        self.assertEqual(report.unvisited_states, ["0"])

    def test_the_target_is_learned(self):
        oracle_creator = lambda nm, s, _d=_target(): NoisyOracle(DFAOracle(_d), nm, s)
        dfa = learn_dfa(oracle_creator, min_signal_strength=0.3, seed=0)
        assertDFA(self, dfa, oracle_creator, sampler=DEFAULT_SAMPLER)


if __name__ == "__main__":
    unittest.main()
