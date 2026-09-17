import unittest

from orthogonal_dfa.l_star.edge_resolver import EdgeResolver
from orthogonal_dfa.l_star.partial_dfa import PartialDFA


class _StubTree:
    def path_of(self, state):
        return (state,)


class _FirstByteSifter:
    """``member + c`` sifts to the member's first byte."""

    def __init__(self):
        self.tree = _StubTree()

    def sift_and_boundary(self, seq):
        return seq[0], None


class _Population:
    def __init__(self, members):
        self.held = list(members)

    def members(self, _path, _limit):
        return list(self.held)


def _resolver(members):
    partial = PartialDFA(alphabet_size=1, num_states=2)
    population = _Population(members)
    return (
        partial,
        population,
        EdgeResolver(partial, _FirstByteSifter(), set(), population=population),
    )


class TestEdgeResolutionMajority(unittest.TestCase):
    def test_majority_wins_over_first_member(self):
        partial, _, resolver = _resolver([bytes([1, 0]), bytes([0, 1]), bytes([0, 2])])
        resolver.close()
        self.assertEqual(partial.target(0, 0), 0)
        self.assertEqual(partial.witness(0, 0), bytes([0, 1]))

    def test_close_flips_an_edge_when_new_members_move_its_majority(self):
        partial, population, resolver = _resolver([bytes([1, 0])])
        resolver.close()
        self.assertEqual(partial.target(0, 0), 1)

        population.held += [bytes([0, 1]), bytes([0, 2])]
        resolver.close()
        self.assertEqual(partial.target(0, 0), 0)
        self.assertEqual(partial.witness(0, 0), bytes([0, 1]))

    def test_tie_keeps_the_current_target(self):
        partial, population, resolver = _resolver([bytes([1, 0])])
        resolver.close()
        population.held.insert(0, bytes([0, 1]))
        resolver.close()
        self.assertEqual(partial.target(0, 0), 1)


if __name__ == "__main__":
    unittest.main()
