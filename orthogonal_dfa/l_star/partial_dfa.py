"""
The partial transition function the resolver builds alongside its tree.

This model owns
    - edges resolved so far
    - queue of edges still to resolve

The edges pointing at a state are not indexed; a split needs them only ~once per
state, so they are scanned out of ``transitions`` on demand instead of being kept
in sync on every ``set_edge``.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple


class PartialDFA:
    def __init__(self, alphabet_size: int, *, num_states: int):
        self.alphabet_size = alphabet_size
        #: transitions[s][c] = s'
        self.transitions: Dict[int, Dict[int, int]] = {s: {} for s in range(num_states)}
        self.worklist: "deque[Tuple[int, int]]" = deque()

    def target(self, state: int, c: int) -> Optional[int]:
        return self.transitions[state].get(c)

    def has_edge(self, state: int, c: int) -> bool:
        return c in self.transitions[state]

    def set_edge(self, state: int, c: int, target: int) -> None:
        self.transitions[state][c] = target

    def reopen(self, state: int, c: int) -> None:
        self.worklist.append((state, c))

    def open_every_edge(self, states) -> None:
        for state in states:
            for c in range(self.alphabet_size):
                self.reopen(state, c)

    def edges_into(self, state: int) -> List[Tuple[int, int]]:
        """The edges whose current target is ``state``, scanned from
        ``transitions`` rather than a maintained reverse index."""
        return [
            (s, c)
            for s, edges in self.transitions.items()
            for c, t in edges.items()
            if t == state
        ]

    def split_state(self, state: int, new_state: int) -> None:
        """Account for ``state`` bifurcating into ``state`` and ``new_state``.

        Both its outgoing edges (computed under the old, larger leaf) and every
        edge into it (which may now belong to either side) are dropped and
        re-queued; then the new leaf's edges are opened.
        """
        self.transitions[new_state] = {}
        for c in range(self.alphabet_size):
            self.transitions[state].pop(c, None)
            self.reopen(state, c)
        for src, c in self.edges_into(state):
            self.transitions[src].pop(c, None)
            self.reopen(src, c)
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
