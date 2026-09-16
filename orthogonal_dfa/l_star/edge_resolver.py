"""
Deciding where the partial DFA's edges point.

PartialDFA owns the edges and the witnesses, but cannot decide where an
edge *goes*, because that needs the oracle.

Every member of the source state votes for where its successor under the
edge's character goes, and the edge points at the majority target, with a member
that voted for it as the witness.  Successors the family cannot place are
harvested as boundary strings; if none can be placed, the edge stays open.

Leaves gain members as the run goes on, so an edge is re-voted on every close
and flips if its majority has moved.
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
        # successor -> the leaf it sifted to, or None if indecisive
        self._sifted: Dict[bytes, Optional[int]] = {}

    def leaf_members(self, state: int) -> List[bytes]:
        return self._population.members(self.sifter.tree.path_of(state), _MEMBER_LIMIT)

    def _sift(self, successor: bytes) -> Optional[int]:
        if successor not in self._sifted:
            target, boundary = self.sifter.sift_and_boundary(successor)
            if target is None:
                self.indecisive.add(boundary)
            self._sifted[successor] = target
        return self._sifted[successor]

    def decisive_target(
        self, state: int, c: int
    ) -> Tuple[Optional[int], Optional[bytes]]:
        votes: Dict[int, List[bytes]] = {}
        for member in self.leaf_members(state):
            target = self._sift(member + bytes([c]))
            if target is not None:
                votes.setdefault(target, []).append(member)
        if not votes:
            return None, None
        # Ties keep the current target, so an edge does not flap between them.
        current = self.dfa.target(state, c)
        target = max(votes, key=lambda t: (len(votes[t]), t == current))
        return target, votes[target][0]

    def split_state(self, state: int, new_state: int) -> None:
        self.dfa.split_state(state, new_state)
        # An indecisive sift stopped above the split leaf, so only these can move.
        self._sifted = {s: t for s, t in self._sifted.items() if t != state}

    def close(self) -> Dict[Tuple[int, int], int]:
        """Re-vote every edge, returning the new target of each edge that changed."""
        edges = [
            (state, c)
            for state in self.dfa.transitions
            for c in range(self.dfa.alphabet_size)
        ]
        changed = {}
        for state, c in track(edges, "Closing edges"):
            target, witness = self.decisive_target(state, c)
            if target is None or target == self.dfa.target(state, c):
                continue
            self.dfa.clear_edge(state, c)
            self.dfa.set_edge(state, c, target, witness)
            changed[state, c] = target
        return changed
