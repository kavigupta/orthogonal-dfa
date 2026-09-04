"""Stopping on a target whose prefixes have nothing left to say.

E-L* names a state by the prefixes that end in it.  Where the language turns on
a state no sampled prefix reaches, no amount of sampling recovers it, and the
round after says exactly what the round before said.
"""

import unittest

from automata.fa.dfa import DFA

from orthogonal_dfa.l_star import preconditions as P
from orthogonal_dfa.l_star.examples.benchmark_generator import DFAOracle
from orthogonal_dfa.l_star.learn import learn_dfa
from orthogonal_dfa.l_star.structures import NoisyOracle
from tests.lstar_common import assert_terminates

#: Residues each branch accepts on.  They differ, so the first symbol decides
#: the language; they overlap, so a suffix that leaves the residue alone
#: preserves the classes and an accept-preserving family still exists.
MOD, ACCEPT_AFTER_ZERO, ACCEPT_AFTER_ONE = 5, {1, 3, 4}, {1, 4}


def _target() -> DFA:
    """``w`` is accepted on ``#1s in w[1:] mod 5``, by a rule ``w[0]`` picks.

    State 0 is left on the first symbol and never re-entered, so the only
    string ending there is the empty one: nothing a length-40 sampler draws
    can say which rule is in force.
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
    def test_the_target_is_learnable_only_up_to_a_coin_flip(self):
        report = P.satisfies_preconditions(_target(), length=40, short_circuit=False)

        self.assertLess(report.covered_accuracy_ceiling, 0.55, report.reasons)
        self.assertEqual(report.uncovered_states, ["0"])
        # The other two preconditions hold, so nothing else stops the run: the
        # language is balanced, and an accept-preserving family exists.
        self.assertGreater(report.class_preserving_fraction, 0.02)
        self.assertGreater(min(report.acceptance_rate, 1 - report.acceptance_rate), 0.2)

    def test_synthesis_stops_rather_than_resampling_forever(self):
        # Every round meets the FNR gate, resolves the same ten states, and
        # harvests a fresh tail of boundary strings that resolve nothing.
        # Counting that tail as progress is what ran this forever.
        oracle_creator = lambda nm, s, _d=_target(): NoisyOracle(DFAOracle(_d), nm, s)

        assert_terminates(
            lambda: learn_dfa(oracle_creator, min_signal_strength=0.3, seed=0),
            seconds=300,
            message="synthesis did not stop on a saturated tree",
        )


if __name__ == "__main__":
    unittest.main()
