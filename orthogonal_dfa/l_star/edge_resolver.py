"""
Deciding where the partial DFA's edges point.

PartialDFA owns the edges and the witnesses, but cannot decide where an
edge *goes*, because that needs the oracle.

Every member of the source state votes for where its successor under the
edge's character goes, and the edge points at the majority target, with a member
that voted for it as the witness.  Successors the family cannot place are
harvested as boundary strings; if none can be placed, the edge stays open.
Leaves gain members as the run goes on, so closing re-votes every edge.
"""

from typing import Dict, List, Optional, Tuple

from .progress import track
from .split_evidence import _MEMBER_LIMIT


class EdgeResolver:
    """Closes the hypothesis: see the module docstring."""

    def __init__(self, partial, sifter, indecisive, *, population):
        self.dfa = partial
        self.sifter = sifter
        self.indecisive = indecisive
        self._population = population

    def leaf_members(self, state: int) -> List[bytes]:
        return self._population.members(self.sifter.tree.path_of(state), _MEMBER_LIMIT)

    def decisive_target(
        self, state: int, c: int
    ) -> Tuple[Optional[int], Optional[bytes]]:
        votes: Dict[int, List[bytes]] = {}
        for member in self.leaf_members(state):
            target, boundary = self.sifter.sift_and_boundary(member + bytes([c]))
            if target is None:
                self.indecisive.add(boundary)
            else:
                votes.setdefault(target, []).append(member)
        if not votes:
            return None, None
        # Ties keep the current target, so an edge does not flap between them.
        current = self.dfa.target(state, c)
        target = max(votes, key=lambda t: (len(votes[t]), t == current))
        return target, votes[target][0]

    def resolve(self, state: int, c: int) -> None:
        target, witness = self.decisive_target(state, c)
        if target is not None and target != self.dfa.target(state, c):
            self.dfa.clear_edge(state, c)
            self.dfa.set_edge(state, c, target, witness)

    def close(self) -> int:
        """
        Re-vote every edge once, returning how many are closed.

        Edge resolution never splits, so one pass resolves all it can; the rest stay
        open for the export to totalise.
        """
        edges = [
            (state, c)
            for state in self.dfa.transitions
            for c in range(self.dfa.alphabet_size)
        ]
        for state, c in track(edges, "Closing edges"):
            self.resolve(state, c)
        return sum(1 for state, c in edges if self.dfa.has_edge(state, c))
