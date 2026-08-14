"""Deciding where the partial DFA's open edges point.

:class:`PartialDFA` owns the edges and the witnesses.  It cannot decide where an
edge *goes*, because that needs the oracle.  This does: it sifts
``member + symbol`` for the members the source leaf already has and takes the
first successor the family can place.

Any member will do -- the tree is consistent, so every member of a leaf resolves
the same edge.  When no member's continuation can be placed the edge is left
open; the export totalises it, and every ``member + symbol`` the family could not
place is harvested into ``indecisive``, which the driver feeds back so the next
round's family resolves it.
"""

from typing import List, Optional, Tuple

from .split_evidence import _MEMBER_LIMIT


class EdgeResolver:
    """Closes the hypothesis: see the module docstring."""

    def __init__(self, partial, sifter, indecisive, *, population):
        self.dfa = partial
        self.sifter = sifter
        self.indecisive = indecisive
        self._population = population

    def leaf_members(self, state: int) -> List[List[int]]:
        return self._population.members(self.sifter.tree.path_of(state), _MEMBER_LIMIT)

    def decisive_target(
        self, state: int, c: int
    ) -> Tuple[Optional[int], Optional[List[int]]]:
        """The first member of ``state`` whose ``c``-successor the family can
        place, and that member; ``(None, None)`` when none can.  Each successor the
        family cannot place is harvested as a boundary string."""
        for member in self.leaf_members(state):
            target, boundary = self.sifter.sift_and_boundary(list(member) + [c])
            if target is not None:
                return target, list(member)
            self.indecisive.add(boundary)
        return None, None

    def resolve(self, state: int, c: int) -> None:
        """Point one edge at a decisive successor, or leave it open."""
        target, witness = self.decisive_target(state, c)
        if target is not None:
            self.dfa.set_edge(state, c, target, witness)

    def close(self) -> int:
        """Resolve every open edge once, returning how many are now closed.  Edge
        resolution never splits, so one pass resolves all it can; the rest stay
        open for the export to totalise."""
        edges = self.dfa.unresolved_edges()
        for state, c in edges:
            self.resolve(state, c)
        return sum(1 for state, c in edges if self.dfa.has_edge(state, c))
