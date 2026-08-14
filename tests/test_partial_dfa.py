import unittest

from orthogonal_dfa.l_star.partial_dfa import PartialDFA


class TestPartialDFA(unittest.TestCase):
    def _dfa(self, num_states=2):
        return PartialDFA(alphabet_size=2, num_states=num_states)

    def test_set_and_read_edge(self):
        d = self._dfa()
        d.set_edge(0, 1, 1, [0, 1])
        self.assertEqual(d.target(0, 1), 1)
        self.assertEqual(d.witness(0, 1), [0, 1])
        self.assertTrue(d.has_edge(0, 1))
        self.assertFalse(d.has_edge(0, 0))

    def test_clear_edge_removes_target_and_witness(self):
        d = self._dfa()
        d.set_edge(0, 1, 1, [3])
        d.clear_edge(0, 1)
        self.assertIsNone(d.target(0, 1))
        self.assertIsNone(d.witness(0, 1))
        self.assertFalse(d.has_edge(0, 1))

    def test_unresolved_edges_shrinks_as_edges_are_set(self):
        d = self._dfa()  # 2 states x 2 symbols = 4 edges
        self.assertEqual(len(d.unresolved_edges()), 4)
        d.set_edge(0, 0, 1, [])
        self.assertEqual(set(d.unresolved_edges()), {(0, 1), (1, 0), (1, 1)})

    def test_split_state_reopens_incident_edges(self):
        d = self._dfa()
        d.set_edge(0, 0, 1, [0])  # into 1
        d.set_edge(1, 0, 0, [1])  # out of 1
        d.split_state(1, 2)
        self.assertFalse(d.has_edge(0, 0))
        self.assertFalse(d.has_edge(1, 0))
        self.assertEqual(d.transitions[2], {})

    def test_totalise_fills_open_edges_and_reports_the_unfillable(self):
        d = self._dfa()
        d.set_edge(0, 0, 1, [])

        def decisive_target(s, c):
            return 0 if (s, c) == (0, 1) else None

        complete, unresolved = d.totalise([0, 1], decisive_target)
        self.assertEqual(complete[0], {0: 1, 1: 0})
        self.assertEqual(complete[1], {0: 1, 1: 1})  # unfillable edges self-loop
        self.assertEqual(sorted(unresolved), [(1, 0), (1, 1)])
        # totalise does not mutate: the edges stay open for a later round
        self.assertEqual(set(d.unresolved_edges()), {(0, 1), (1, 0), (1, 1)})


if __name__ == "__main__":
    unittest.main()
