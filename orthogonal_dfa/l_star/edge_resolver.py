"""
Deciding where the partial DFA's open edges point.

PartialDFA owns the edges and the witnesses, but cannot decide where an
edge *goes*, because that needs the oracle.

We pick an arbitrary member of a source state, and ask the oracle
where its successor under the edge's character goes.

    - If the family can place that successor, we point the edge there and
      record the member as the witness.
    - If the family cannot place that successor, we harvest it as a boundary
      string and leave the edge open.
"""

from typing import List, Optional, Tuple

from .split_evidence import _MEMBER_LIMIT


class EdgeResolver:
    """Closes the hypothesis: see the module docstring."""

    def __init__(self, partial, sifter, indecisive, *, population, decisions):
        self.dfa = partial
        self.sifter = sifter
        self.indecisive = indecisive
        self._population = population
        self._decisions = decisions

    def leaf_members(self, state: int) -> List[bytes]:
        return self._population.members(self.sifter.tree.path_of(state), _MEMBER_LIMIT)

    def decisive_target(
        self, state: int, c: int
    ) -> Tuple[Optional[int], Optional[bytes]]:
        for member in self.leaf_members(state):
            target, boundary = self.sifter.sift_and_boundary(member + bytes([c]))
            self._decisions.record(target is not None)
            if target is not None:
                return target, member
            self.indecisive.add(boundary)
        return None, None

    def resolve(self, state: int, c: int) -> None:
        target, witness = self.decisive_target(state, c)
        if target is not None:
            self.dfa.set_edge(state, c, target, witness)

    def close(self) -> int:
        """
        Resolve every open edge once, returning how many are now closed.

        Edge resolution never splits, so one pass resolves all it can; the rest stay
        open for the export to totalise.
        """
        edges = self.dfa.unresolved_edges()
        for state, c in edges:
            self.resolve(state, c)
        return sum(1 for state, c in edges if self.dfa.has_edge(state, c))
