"""Deciding where the partial DFA's open edges point.

:class:`PartialDFA` owns the edges, the witnesses and the queue.  It cannot
decide where an edge *goes*, because that needs the oracle.  This does: it sifts
``member + symbol`` for members of the source leaf and takes the first successor
the family can place.

Any member will do -- the tree is consistent, so every member of a leaf resolves
the same edge.  That is why it does not stop at the access string: one
indecisive continuation would otherwise strand an edge the leaf as a whole can
place.  Only an edge whose *entire* leaf is indecisive is unresolvable.
"""

from typing import Iterator, List, Optional, Tuple

#: Leaf members to try before giving an edge up as unresolvable.
MAX_EDGE_TRIES = 30


class EdgeResolver:
    """Closes the hypothesis: see the module docstring.

    ``indecisive`` is the sink for boundary strings -- every ``member + symbol``
    the family cannot place is recorded there, because the driver feeds them back
    so the next round's family is forced to resolve them.
    """

    def __init__(self, partial, sifter, indecisive, *, population, representative):
        self.dfa = partial
        self.sifter = sifter
        self.indecisive = indecisive
        self._population = population
        self._representative = representative

    # -- opening the queue ---------------------------------------------------

    def open_all_edges(self) -> None:
        self.dfa.open_every_edge(range(self.sifter.tree.num_states))

    # -- resolving -----------------------------------------------------------

    def leaf_members(self, state: int, *, limit: int) -> List[List[int]]:
        return self._population.members(self.sifter.tree.path_of(state), limit)

    def _candidates(self, state: int) -> Iterator[List[int]]:
        """Members to try, the shortest first, yielded lazily so the leaf's
        population is pulled down only as far as an edge needs."""
        rep = self._representative(state)
        if rep is not None:
            yield rep
        yield from self.leaf_members(state, limit=MAX_EDGE_TRIES)

    def decisive_target(
        self, state: int, c: int
    ) -> Tuple[Optional[int], Optional[List[int]]]:
        """A *decisive* target for ``delta(state, c)``, and the member that gave
        it.  Returns ``(None, None)`` only when every member tried is
        indecisive."""
        seen, tries = set(), 0
        for member in self._candidates(state):
            key = tuple(member)
            if key in seen:
                continue
            seen.add(key)
            target, boundary = self.sifter.sift_and_boundary(list(member) + [c])
            if target is not None:
                return target, list(member)
            # This successor is a boundary string the family cannot place.
            self.indecisive.add(boundary)
            tries += 1
            if tries >= MAX_EDGE_TRIES:
                break
        return None, None

    def resolve(self, state: int, c: int) -> None:
        """Point one edge at a decisive successor, or leave it open."""
        if self._representative(state) is None:
            return  # unreachable leaf; leave the edge for the export fallback
        target, witness = self.decisive_target(state, c)
        if target is None:
            return  # every member indecisive; export fills it as a self-loop
        self.dfa.set_edge(state, c, target, witness)

    def close(self) -> int:
        """Resolve queued edges until the hypothesis is closed.  Returns the
        number resolved."""
        self.sifter.prefill(self.dfa.pending_probes(self._representative))
        return self.dfa.drain(self.resolve)
