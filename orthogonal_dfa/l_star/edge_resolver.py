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

import os

from typing import List, Optional, Tuple

from .split_evidence import _MEMBER_LIMIT

# Opt-in instrumentation: when DLSTAR_MEMBERLOG is set, log how many member
# access-strings backed each undecidable (self-loop-bound) edge, and how many of
# them were individually indecisive.  Answers "is a self-loop a confidently
# unconfident decision, or one made on a handful of members?"
_MEMBERLOG = os.environ.get("DLSTAR_MEMBERLOG")


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
        members = self.leaf_members(state)
        n_indecisive = 0
        for member in members:
            target, boundary = self.sifter.sift_and_boundary(list(member) + [c])
            if target is not None:
                return target, list(member)
            self.indecisive.add(boundary)
            n_indecisive += 1
        if _MEMBERLOG:
            print(
                f"[memberlog] self-loop (state {state}, symbol {c}): "
                f"{len(members)} members, all {n_indecisive} indecisive",
                flush=True,
            )
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
