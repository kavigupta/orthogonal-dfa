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

import random

from typing import List, Optional, Tuple

from .progress import track
from .split_evidence import _MEMBER_LIMIT

#: Members polled to settle one edge's target.  One would do where a leaf is
#: homogeneous, but a leaf is often not: its members' successors sift to more than
#: one target, and then the *first* decisively-sifting member captures the whole
#: edge -- an arbitrary, often minority, choice that whichever member the
#: population happens to yield first decides.  So poll several and take the target
#: the most of them agree on.  Capped and sampled because a leaf can hold up to
#: ``_MEMBER_LIMIT`` members and each vote costs a sift.
_EDGE_VOTES = 24


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
        """The target the most of the leaf's members route ``c`` to.

        A random sample of the members (by position, so the order the population
        yields them in cannot stack a split leaf's vote), voted by majority; the
        witness is a member that routed to the winner.  Where every member agrees
        the majority is that agreement, so a homogeneous leaf resolves as before.
        Falls back to a full scan only if the sample sifts nowhere, so an edge a
        member outside the sample could still settle is not left open by sampling.
        """
        members = self.leaf_members(state)
        sample = members
        if len(members) > _EDGE_VOTES:
            sample = random.Random(state * 1_000_003 + c).sample(members, _EDGE_VOTES)
        target, witness = self._vote(sample, c)
        if target is None and sample is not members:
            return self._first_decisive(members, c)
        return target, witness

    def _vote(self, members, c):
        counts: dict = {}
        witness: dict = {}
        decided = False
        for member in members:
            target, boundary = self.sifter.sift_and_boundary(member + bytes([c]))
            if target is not None:
                counts[target] = counts.get(target, 0) + 1
                witness.setdefault(target, member)
                decided = True
            elif not decided:
                self.indecisive.add(boundary)
        if not counts:
            return None, None
        best = max(counts, key=lambda t: counts[t])
        return best, witness[best]

    def _first_decisive(self, members, c):
        for member in members:
            target, boundary = self.sifter.sift_and_boundary(member + bytes([c]))
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
        for state, c in track(edges, "Closing edges"):
            self.resolve(state, c)
        return sum(1 for state, c in edges if self.dfa.has_edge(state, c))
