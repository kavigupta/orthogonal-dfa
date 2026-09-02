import signal
import unittest

from orthogonal_dfa.l_star.edge_resolver import EdgeResolver
from orthogonal_dfa.l_star.partial_dfa import PartialDFA


class _StubTree:
    def path_of(self, state):
        return (state,)


class _AlwaysIndecisiveSifter:
    """A sifter whose confident sift can never place a string -- every
    ``member + symbol`` sifts to ``None``, harvesting the boundary -- but whose
    decisive fallback always lands a leaf, as :meth:`SuffixFamily.side` does."""

    def __init__(self):
        self.tree = _StubTree()

    def prefill(self, probes):
        pass

    def sift_and_boundary(self, seq):
        return None, seq

    def sift_decisive(self, _seq):
        return 0


class _StubPopulation:
    """Every leaf is reachable and has a member, so the resolver actually tries to
    resolve each edge (rather than skipping it as unreachable)."""

    def representative(self, _path, _limit):
        return bytes([0])

    def members(self, _path, _limit):
        return [bytes([0])]


class TestEdgeResolverCloseTerminates(unittest.TestCase):
    def test_close_places_indecisive_edges_by_the_decisive_fallback(self):
        # Regression: close() drained edges until none were missing, but resolve()
        # left an edge open when the whole leaf was indecisive -- and an open edge
        # is still "missing", so the drain retried it forever (a hang that surfaced
        # only under a different numpy float path). close() must be a single pass.
        # The decisive fallback now places such an edge instead of leaving it open,
        # so nothing is missing -- while the confident sift still harvests each
        # boundary for the next round.
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

        # The confident sift harvested every boundary; the decisive fallback then
        # placed every edge, so none is left open and close() returned in one pass.
        self.assertEqual(resolved, 4)
        self.assertEqual(partial.unresolved_edges(), [])
        self.assertTrue(resolver.indecisive)

    @staticmethod
    def _timeout(signum, frame):
        raise AssertionError(
            "EdgeResolver.close did not terminate (drain spun on an open edge)"
        )


if __name__ == "__main__":
    unittest.main()
