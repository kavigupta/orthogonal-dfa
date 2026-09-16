import signal
import unittest

from orthogonal_dfa.l_star.edge_resolver import EdgeResolver
from orthogonal_dfa.l_star.partial_dfa import PartialDFA


class _StubTree:
    def path_of(self, state):
        return (state,)


class _AlwaysIndecisiveSifter:
    """A sifter that can never place a string: every ``member + symbol`` sifts to
    ``None``, so every edge the resolver tries is left open."""

    def __init__(self):
        self.tree = _StubTree()

    def prefill(self, probes):
        pass

    def sift_and_boundary(self, seq):
        return None, seq


class _StubPopulation:
    """Every leaf is reachable and has a member, so the resolver actually tries to
    resolve each edge (rather than skipping it as unreachable)."""

    def representative(self, _path, _limit):
        return bytes([0])

    def members(self, _path, _limit):
        return [bytes([0])]


class TestEdgeResolverCloseTerminates(unittest.TestCase):
    def test_close_is_single_pass_when_every_edge_is_left_open(self):
        # Regression: close() drained edges until none were missing, but resolve()
        # leaves an edge open when the whole leaf is indecisive -- and an open edge
        # is still "missing", so the drain retried it forever (a hang that surfaced
        # only under a different numpy float path). close() must be a single pass.
        partial = PartialDFA(alphabet_size=2, num_states=2)
        resolver = EdgeResolver(
            partial,
            _AlwaysIndecisiveSifter(),
            set(),
            population=_StubPopulation(),
        )

        previous = signal.signal(signal.SIGALRM, self._timeout)
        signal.alarm(5)
        try:
            changed = resolver.close()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)

        # Nothing could be placed, so every edge is left open for the export
        # fallback -- and, crucially, close() returned instead of spinning.
        self.assertEqual(changed, {})
        self.assertEqual(partial.unresolved_edges(), [(0, 0), (0, 1), (1, 0), (1, 1)])
        self.assertTrue(resolver.indecisive)

    @staticmethod
    def _timeout(signum, frame):
        raise AssertionError(
            "EdgeResolver.close did not terminate (drain spun on an open edge)"
        )


class _GrowingPopulation:
    def __init__(self, members):
        self.held = list(members)

    def members(self, _path, _limit):
        return list(self.held)


class _FirstByteSifter:
    """``member + c`` sifts to the member's first byte."""

    def __init__(self):
        self.tree = _StubTree()

    def sift_and_boundary(self, seq):
        return seq[0], None


class TestEdgeResolverRevotes(unittest.TestCase):
    def test_close_flips_an_edge_when_new_members_move_its_majority(self):
        partial = PartialDFA(alphabet_size=1, num_states=2)
        population = _GrowingPopulation([bytes([1, 0])])
        resolver = EdgeResolver(partial, _FirstByteSifter(), set(), population=population)
        resolver.close()
        self.assertEqual(partial.target(0, 0), 1)

        population.held += [bytes([0, 1]), bytes([0, 2])]
        self.assertEqual(resolver.close(), {(0, 0): 0, (1, 0): 0})
        self.assertEqual(partial.target(0, 0), 0)
        self.assertEqual(partial.witness(0, 0), bytes([0, 1]))


if __name__ == "__main__":
    unittest.main()
