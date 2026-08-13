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

    def test_drain_resolves_every_edge_and_counts_them(self):
        d = self._dfa()
        resolved = []

        def resolve(s, c):
            resolved.append((s, c))
            d.set_edge(s, c, 0, [s])

        self.assertEqual(d.drain(resolve), 4)
        self.assertEqual(d.unresolved_edges(), [])

    def test_drain_loops_until_no_edge_is_missing(self):
        # A resolve that splits on its first call reopens edges; drain must pick
        # them up rather than stopping at the initial set.
        d = PartialDFA(alphabet_size=2, num_states=1)
        calls = []

        def resolve(s, c):
            calls.append((s, c))
            d.set_edge(s, c, 0, [])
            if len(calls) == 1:
                d.split_state(0, 1)

        count = d.drain(resolve)
        self.assertEqual(d.unresolved_edges(), [])
        self.assertGreater(count, 2)  # more than the initial 2 edges, due to the split

    def test_pending_probes_extends_each_representative_by_its_symbol(self):
        d = self._dfa()
        d.set_edge(0, 0, 1, [])  # leaves (0,1), (1,0), (1,1) open
        reps = {0: [9], 1: [8, 8]}
        probes = d.pending_probes(lambda s: reps[s])
        self.assertEqual(sorted(probes), sorted([[9, 1], [8, 8, 0], [8, 8, 1]]))

    def test_pending_probes_skips_edges_without_a_representative(self):
        d = self._dfa()
        probes = d.pending_probes(lambda s: [s] if s == 0 else None)
        self.assertEqual(sorted(probes), sorted([[0, 0], [0, 1]]))

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
