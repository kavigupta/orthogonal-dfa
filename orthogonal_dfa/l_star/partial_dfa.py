"""
The partial transition function the resolver builds alongside its tree.

Also contains methods for splitting a state and for totalising the partial delta
into a complete transition function for export.
"""

from typing import Dict, List, Optional, Tuple


class PartialDFA:
    def __init__(self, alphabet_size: int, *, num_states: int):
        self.alphabet_size = alphabet_size
        #: transitions[s][c] = s'
        self.transitions: Dict[int, Dict[int, int]] = {s: {} for s in range(num_states)}
        #: witnesses[s, c] = A prefix x such that dfa(x) = s and dfa(x + [c]) = transitions[s][c]
        self.witnesses: Dict[Tuple[int, int], List[int]] = {}

    def target(self, state: int, c: int) -> Optional[int]:
        return self.transitions[state].get(c)

    def has_edge(self, state: int, c: int) -> bool:
        return c in self.transitions[state]

    def witness(self, state: int, c: int) -> Optional[List[int]]:
        return self.witnesses.get((state, c))

    def set_edge(self, state: int, c: int, target: int, witness) -> None:
        self.transitions[state][c] = target
        self.witnesses[state, c] = list(witness)

    def clear_edge(self, state: int, c: int) -> None:
        self.transitions[state].pop(c, None)
        self.witnesses.pop((state, c), None)

    def edges_into(self, state: int) -> List[Tuple[int, int]]:
        """The edges whose current target is ``state``, scanned from
        ``transitions`` rather than a maintained reverse index."""
        return [
            (s, c)
            for s, edges in self.transitions.items()
            for c, t in edges.items()
            if t == state
        ]

    def unresolved_edges(self) -> List[Tuple[int, int]]:
        """Every edge still missing from ``transitions``."""
        return [
            (s, c)
            for s, edges in self.transitions.items()
            for c in range(self.alphabet_size)
            if c not in edges
        ]

    def split_state(self, state: int, new_state: int) -> None:
        """
        Account for state bifurcating into (state, new_state).

        All relevant edges are removed, both outgoing from and incoming to state.
        """
        self.transitions[new_state] = {}
        for c in range(self.alphabet_size):
            self.clear_edge(state, c)
        for src, c in self.edges_into(state):
            self.clear_edge(src, c)

    def totalise(self, states, decisive_target):
        """
        A "total" copy this partial DFA. An open edge is filled from
            decisive_target(state, symbol),
        or self-looped and reported in the second return value where that
        fails too.

        Does not mutate transitions, so a later round can still close the open edges.
        """
        complete: Dict[int, Dict[int, int]] = {}
        unresolved: List[Tuple[int, int]] = []
        for state in states:
            complete[state] = dict(self.transitions[state])
            for c in range(self.alphabet_size):
                if c in complete[state]:
                    continue
                target = decisive_target(state, c)
                if target is None:
                    unresolved.append((state, c))
                    target = state
                complete[state][c] = target
        return complete, unresolved
