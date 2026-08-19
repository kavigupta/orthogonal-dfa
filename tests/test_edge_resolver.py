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
        return None, tuple(seq)


class _StubPopulation:
    """Every leaf is reachable and has a member, so the resolver actually tries to
    resolve each edge (rather than skipping it as unreachable)."""

    def representative(self, _path, _limit):
        return [0]

    def members(self, _path, _limit):
        return [[0]]


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
            resolved = resolver.close()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)

        # Nothing could be placed, so every edge is left open for the export
        # fallback -- and, crucially, close() returned instead of spinning.
        self.assertEqual(resolved, 0)
        self.assertEqual(partial.unresolved_edges(), [(0, 0), (0, 1), (1, 0), (1, 1)])
        self.assertTrue(resolver.indecisive)

    @staticmethod
    def _timeout(signum, frame):
        raise AssertionError(
            "EdgeResolver.close did not terminate (drain spun on an open edge)"
        )


if __name__ == "__main__":
    unittest.main()
