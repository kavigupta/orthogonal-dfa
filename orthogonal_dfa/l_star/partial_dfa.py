"""
The partial transition function the resolver builds alongside its tree.

Also contains methods for splitting states and a loop for resolving unresolved edges.
"""

from typing import Dict, List, Optional, Tuple


class PartialDFA:
    def __init__(self, alphabet_size: int, *, num_states: int):
        self.alphabet_size = alphabet_size
        #: transitions[s][c] = s'
        self.transitions: Dict[int, Dict[int, int]] = {s: {} for s in range(num_states)}

    def target(self, state: int, c: int) -> Optional[int]:
        return self.transitions[state].get(c)

    def has_edge(self, state: int, c: int) -> bool:
        return c in self.transitions[state]

    def set_edge(self, state: int, c: int, target: int) -> None:
        self.transitions[state][c] = target

    def edges_into(self, state: int) -> List[Tuple[int, int]]:
        """The edges whose current target is ``state``, scanned from
        ``transitions`` rather than a maintained reverse index."""
        return [
            (s, c)
            for s, edges in self.transitions.items()
            for c, t in edges.items()
            if t == state
        ]

    def _next_unresolved(self) -> Optional[Tuple[int, int]]:
        for state, edges in self.transitions.items():
            for c in range(self.alphabet_size):
                if c not in edges:
                    return (state, c)
        return None

    def split_state(self, state: int, new_state: int) -> None:
        """Account for ``state`` bifurcating into ``state`` and ``new_state``.

        Both its outgoing edges (computed under the old, larger leaf) and every
        edge into it (which may now belong to either side) are dropped; they and
        the new leaf's edges then read as unresolved and get re-resolved.
        """
        self.transitions[new_state] = {}
        for c in range(self.alphabet_size):
            self.transitions[state].pop(c, None)
        for src, c in self.edges_into(state):
            self.transitions[src].pop(c, None)

    def drain(self, resolve) -> None:
        """Resolve edges via ``resolve(state, symbol)`` until every one is filled.
        ``resolve`` may split -- clearing edges, which the next scan picks back
        up -- so this loops until no edge is missing."""
        while (edge := self._next_unresolved()) is not None:
            resolve(*edge)
