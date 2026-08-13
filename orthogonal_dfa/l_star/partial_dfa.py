"""The partial transition function the resolver builds alongside its tree.

It owns the edges resolved so far, the set of edges pointing at each state, and
the queue of edges still to resolve.  Keeping this here lets a split invalidate
*precisely* the edges it made ambiguous -- the leaf's own outgoing edges and
every edge into it -- instead of rebuilding the hypothesis: an edge that never
touched the split leaf keeps its resolved target.
"""

from collections import deque
from typing import Dict, Optional, Set, Tuple


class PartialDFA:
    def __init__(self, alphabet_size: int, *, num_states: int):
        self.alphabet_size = alphabet_size
        #: ``transitions[s][c]`` -- the resolved target of edge ``(s, c)``.
        self.transitions: Dict[int, Dict[int, int]] = {s: {} for s in range(num_states)}
        #: ``incoming[s]`` -- the edges whose target is ``s``, so a split can
        #: re-open exactly those.
        self.incoming: Dict[int, Set[Tuple[int, int]]] = {
            s: set() for s in range(num_states)
        }
        self.worklist: "deque[Tuple[int, int]]" = deque()

    def target(self, state: int, c: int) -> Optional[int]:
        return self.transitions[state].get(c)

    def has_edge(self, state: int, c: int) -> bool:
        return c in self.transitions[state]

    def set_edge(self, state: int, c: int, target: int) -> None:
        previous = self.transitions[state].get(c)
        if previous is not None:
            self.incoming[previous].discard((state, c))
        self.transitions[state][c] = target
        self.incoming[target].add((state, c))

    def clear_edge(self, state: int, c: int) -> None:
        target = self.transitions[state].pop(c, None)
        if target is not None:
            self.incoming[target].discard((state, c))

    def reopen(self, state: int, c: int) -> None:
        self.worklist.append((state, c))

    def open_every_edge(self, states) -> None:
        for state in states:
            for c in range(self.alphabet_size):
                self.reopen(state, c)

    def split_state(self, state: int, new_state: int) -> None:
        """Account for ``state`` bifurcating into ``state`` and ``new_state``.

        Both its outgoing edges (computed under the old, larger leaf) and every
        edge into it (which may now belong to either side) are dropped and
        re-queued; then the new leaf's edges are opened.
        """
        self.transitions[new_state] = {}
        self.incoming[new_state] = set()
        for c in range(self.alphabet_size):
            self.clear_edge(state, c)
            self.reopen(state, c)
        for src, c in list(self.incoming[state]):
            self.clear_edge(src, c)
            self.reopen(src, c)
        self.incoming[state] = set()
        self.open_every_edge((new_state,))

    def drain(self, resolve) -> None:
        """Resolve queued edges via ``resolve(state, symbol)`` until the
        hypothesis is closed.  A split reuses the old id, so ids stay dense and the
        only dedup is skipping edges a later step already resolved."""
        while self.worklist:
            state, c = self.worklist.popleft()
            if self.has_edge(state, c):
                continue
            resolve(state, c)
