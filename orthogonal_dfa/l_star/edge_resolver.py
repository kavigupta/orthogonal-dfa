"""Deciding where the partial DFA's open edges point.

:class:`PartialDFA` owns the edges, the witnesses and the queue.  It cannot
decide where an edge *goes*, because that needs the oracle.  This does: it sifts
``member + symbol`` for members of the source leaf and takes the first successor
the family can place.

Any member will do -- the tree is consistent, so every member of a leaf resolves
the same edge -- which is the whole point of not stopping at the access string.
An earlier version sifted only ``access[state] + [symbol]`` and left the edge to
a self-loop when that one string happened to be indecisive, wrecking the exported
DFA even when the tree was correct.  Only an edge whose *entire* leaf is
indecisive is genuinely unresolvable.
"""

from typing import List, Optional, Tuple

#: Pool prefixes a leaf-membership scan sifts per batched pass.
MEMBER_SCAN_BLOCK = 128
#: Leaf members to try before giving an edge up as unresolvable.
MAX_EDGE_TRIES = 30


class EdgeResolver:
    """Closes the hypothesis: see the module docstring.

    ``indecisive`` is the sink for boundary strings -- every ``member + symbol``
    the family cannot place is recorded there, because the driver feeds them back
    so the next round's family is forced to resolve them.
    """

    def __init__(self, pst, partial, sifter, indecisive, *, max_tries: int):
        self.pst = pst
        self.dfa = partial
        self.sifter = sifter
        self.indecisive = indecisive
        self._max_tries = max_tries

    # -- opening the queue ---------------------------------------------------

    def seed_access(self) -> None:
        """Give every current leaf a canonical access string by sifting the pool.
        The empty string pins the initial state; the rest come from whatever pool
        prefixes land in each leaf."""
        states = self.sifter.tree.num_states
        for prefix in [[]] + [list(p) for p in self.pst.table.prefixes]:
            if len(self.dfa.access) >= states:
                break
            state = self.sifter.sift(prefix)
            if state is not None and state not in self.dfa.access:
                self.dfa.access[state] = list(prefix)

    def open_all_edges(self) -> None:
        self.seed_access()
        self.dfa.open_every_edge(range(self.sifter.tree.num_states))

    # -- resolving -----------------------------------------------------------

    def leaf_members(
        self, state: int, *, limit: Optional[int] = None
    ) -> List[List[int]]:
        """Pool prefixes that sift to ``state``."""
        return self.sifter.leaves_of(
            [list(p) for p in self.pst.table.prefixes],
            state,
            limit=limit,
            block=MEMBER_SCAN_BLOCK,
        )

    def find_access(self, state: int) -> Optional[List[int]]:
        cached = self.dfa.access.get(state)
        if cached is not None:
            return cached
        for prefix in self.pst.table.prefixes:
            if self.sifter.sift(list(prefix)) == state:
                self.dfa.access[state] = list(prefix)
                return list(prefix)
        return None

    def decisive_target(
        self, state: int, c: int
    ) -> Tuple[Optional[int], Optional[List[int]]]:
        """A *decisive* target for ``delta(state, c)``, and the member that gave
        it.  Tries the access string first, then other leaf members; returns
        ``(None, None)`` only when every one tried is indecisive."""
        candidates: List[List[int]] = []
        access = self.dfa.access.get(state)
        if access is not None:
            candidates.append(access)
        candidates.extend(self.leaf_members(state, limit=self._max_tries))
        seen, tries = set(), 0
        for member in candidates:
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
            if tries >= self._max_tries:
                break
        return None, None

    def resolve(self, state: int, c: int) -> None:
        """Point one edge at a decisive successor, or leave it open."""
        if self.dfa.access.get(state) is None and self.find_access(state) is None:
            return  # unreachable leaf; leave the edge for the export fallback
        target, witness = self.decisive_target(state, c)
        if target is None:
            return  # every member indecisive; export fills it as a self-loop
        self.dfa.set_edge(state, c, target, witness)

    def close(self) -> int:
        """Resolve queued edges until the hypothesis is closed.  Returns the
        number resolved."""
        self.sifter.prefill(self.dfa.pending_probes())
        return self.dfa.drain(self.resolve)
