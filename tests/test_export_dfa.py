import unittest
from types import SimpleNamespace

from orthogonal_dfa.l_star.direct_lstar import DirectLStarLearner, export_dfa
from orthogonal_dfa.l_star.midfix_tree import MidfixTree
from orthogonal_dfa.l_star.partial_dfa import PartialDFA
from tests.direct_lstar_stubs import make_pst


def _family():
    return SimpleNamespace(vs=[0, 1])


class TestExportDfa(unittest.TestCase):
    def test_open_edges_are_filled_from_decisive_target(self):
        # Regression: export reaches decisive_target as a *callback*, so a
        # signature change that the direct call sites absorb can still break here
        # -- and only for a hypothesis the worklist left incomplete, which the
        # easy benchmarks never produce.
        tree = MidfixTree()
        partial = PartialDFA(2, num_states=tree.num_states)
        partial.set_edge(0, 0, 0, witness=[])
        # (0,1), (1,0) and (1,1) are left open on purpose.
        asked = []

        def decisive_target(state, c):
            asked.append((state, c))
            return 1

        dfa, _ = export_dfa(tree, partial, _family(), make_pst(), decisive_target)
        self.assertEqual([(0, 1), (1, 0), (1, 1)], sorted(asked))
        self.assertEqual({0: {0: 0, 1: 1}, 1: {0: 1, 1: 1}}, dfa.transitions)

    def test_an_unresolvable_edge_self_loops(self):
        # Only an edge whose entire leaf is indecisive should fall back, and it
        # must still produce a total transition function.
        tree = MidfixTree()
        partial = PartialDFA(2, num_states=tree.num_states)
        dfa, _ = export_dfa(tree, partial, _family(), make_pst(), lambda state, c: None)
        self.assertEqual({0: {0: 0, 1: 0}, 1: {0: 1, 1: 1}}, dfa.transitions)

    def test_a_closed_hypothesis_never_asks(self):
        tree = MidfixTree()
        partial = PartialDFA(2, num_states=tree.num_states)
        for state in (0, 1):
            for c in (0, 1):
                partial.set_edge(state, c, state, witness=[])

        def decisive_target(state, c):
            raise AssertionError(f"asked about the resolved edge ({state}, {c})")

        dfa, _ = export_dfa(tree, partial, _family(), make_pst(), decisive_target)
        self.assertEqual({0: {0: 0, 1: 0}, 1: {0: 1, 1: 1}}, dfa.transitions)


class TestLearnerExport(unittest.TestCase):
    def test_exporting_an_incomplete_hypothesis(self):
        # Regression: the learner hands export_dfa its edge resolver as a
        # *callback*, so an arity change that every direct call site absorbs can
        # still break here -- and only when the worklist left an edge open, which
        # the quick benchmarks never do.
        learner = DirectLStarLearner(
            make_pst(), [0, 1], split_fpr=None, split_miss_rate=0.02
        )
        learner.population.add([], at=learner.tree.path_of(0))
        learner.population.add([1], at=learner.tree.path_of(1))
        dfa, _ = learner.to_dfa_and_tree()  # every edge still open
        for state in dfa.states:
            self.assertEqual({0, 1}, set(dfa.transitions[state]))


if __name__ == "__main__":
    unittest.main()
