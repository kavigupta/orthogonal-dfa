"""The ladder gate: bound a non-merging prepend-ladder without touching a target
whose splits merge back into the automaton.

The unit tests pin the merge signal directly (fast, deterministic); the integration
test checks the gate does not fire on a regular target -- the safety property, since
the gate is otherwise dead code (only active when ``ladder_budget`` is passed)."""

import unittest
from types import SimpleNamespace

from orthogonal_dfa.l_star.direct_lstar import DirectLStarLearner
from orthogonal_dfa.l_star.partial_dfa import PartialDFA


class _GateLearner(DirectLStarLearner):
    """A learner reduced to just what ``_merges_into_existing`` reads."""

    # pylint: disable=super-init-not-called
    def __init__(self, num_states, alphabet_size=4):
        self.pst = SimpleNamespace(alphabet_size=alphabet_size)
        self.dfa = PartialDFA(alphabet_size, num_states=num_states)


class TestMergeSignal(unittest.TestCase):
    def test_back_edge_to_an_older_state_merges(self):
        learner = _GateLearner(3)
        # new state 2 resolves an edge back to the older state 0.
        learner.dfa.set_edge(2, 0, 0, witness=[1])
        self.assertTrue(learner._merges_into_existing(2))

    def test_incoming_from_an_older_state_merges(self):
        learner = _GateLearner(3)
        # an older state 1 now transitions into the new state 2.
        learner.dfa.set_edge(1, 0, 2, witness=[0])
        self.assertTrue(learner._merges_into_existing(2))

    def test_only_forward_or_self_edges_do_not_merge(self):
        learner = _GateLearner(3)
        # a shift-register rung: its one resolved edge is a self-loop, and nothing
        # older points into it -- it flows forward, never closing back.
        learner.dfa.set_edge(2, 0, 2, witness=[1])
        self.assertFalse(learner._merges_into_existing(2))

    def test_no_resolved_edges_do_not_merge(self):
        learner = _GateLearner(3)
        self.assertFalse(learner._merges_into_existing(2))


class TestGateSafeOnRegularTarget(unittest.TestCase):
    def test_regular_target_unaffected_by_the_gate(self):
        """Parity's splits close and merge, so a tight ladder budget never trips:
        the learner still recovers the compact automaton at high accuracy."""
        import numpy as np

        from orthogonal_dfa.analysis.hidden_signal import ParityOracle
        from orthogonal_dfa.l_star.counterexample_synthesis import (
            synthesize_direct_lstar_fnr,
        )
        from orthogonal_dfa.l_star.learn import build_pst

        length = 12
        oracle = ParityOracle()
        pst = build_pst(
            lambda _n, _s: oracle,
            min_signal_strength=0.06,
            seed=0,
            sample_length=length,
            fnr_limit=0.02,
        )
        dfa, _ = synthesize_direct_lstar_fnr(
            pst,
            acc_threshold=0.98,
            max_rounds=4,
            counterexample_probes=200,
            per_state=12,
            min_indecisive=40,
            ladder_budget=3,
        )
        held = np.random.default_rng(1).integers(0, 4, (2000, length))
        call = np.array([bool(dfa.accepts_input(s.tolist())) for s in held])
        truth = np.array([oracle.membership_query(s.tolist()) for s in held])
        self.assertGreaterEqual(float((call == truth).mean()), 0.95)
        self.assertLessEqual(len(dfa.states), 6)


if __name__ == "__main__":
    unittest.main()
