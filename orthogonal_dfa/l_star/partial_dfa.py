"""The partial transition function the resolver builds alongside its tree.

``delta`` under construction: the edges resolved so far and the witness prefix
that justified each one.  Keeping it here is what lets a split invalidate
*precisely* the edges it made ambiguous instead of rebuilding the hypothesis: an
edge that does not touch the split leaf keeps a valid witness, because its sift
path never passed through that leaf.

The object owns bookkeeping only.  *Deciding* where an edge points needs the
oracle, so the caller supplies ``resolve(state, symbol)`` when closing and
``decisive_target(state, symbol)`` when totalising.  Neither the edges pointing at
a state nor the edges still to resolve are indexed: a split needs the former only
~once per state, and an unresolved edge is just one missing from ``transitions``
-- both are scanned on demand.
"""

from typing import Dict, List, Optional, Tuple


class PartialDFA:
    def __init__(self, alphabet_size: int, *, num_states: int):
        self.alphabet_size = alphabet_size
        #: ``transitions[s][c]`` -- the current best guess for ``delta(s, c)``.
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
        """Account for ``state`` having bifurcated into ``state`` and ``new_state``.

        Only edges incident to the old leaf become ambiguous: its outgoing edges
        vanish (the source is now two states) and every edge into it must be
        re-classified.  Both sets are dropped; they and the new leaf's edges then
        read as unresolved and get re-resolved."""
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
