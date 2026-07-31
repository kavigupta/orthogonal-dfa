"""The partial transition function direct-L* builds alongside its tree.

``delta`` under construction: the edges resolved so far, the witness prefix that
justified each one, a canonical access string per state, and the queue of edges
still to resolve.  Keeping it here is what lets a split invalidate *precisely*
the edges it made ambiguous instead of rebuilding the hypothesis: an edge that
does not touch the split leaf keeps a valid witness, because its sift path never
passed through that leaf.

The object owns bookkeeping only.  *Deciding* where an edge points needs the
oracle, so the caller supplies ``resolve(state, symbol)`` when draining the queue
and ``decisive_target(state, symbol)`` when totalising -- the same division of
labour as :class:`MidfixTree`'s ``decide`` callback.
"""

from collections import deque
from typing import Dict, List, Optional, Set, Tuple


class PartialDFA:
    """See the module docstring."""

    def __init__(self, alphabet_size: int, *, num_states: int):
        self.alphabet_size = alphabet_size
        #: ``transitions[s][c]`` -- the current best guess for ``delta(s, c)``.
        self.transitions: Dict[int, Dict[int, int]] = {s: {} for s in range(num_states)}
        #: A prefix that provably reaches ``s`` and whose one-symbol extension by
        #: ``c`` reaches ``transitions[s][c]``.
        self.witnesses: Dict[Tuple[int, int], List[int]] = {}
        #: A canonical access string per state.  Edges are resolved from it, so it
        #: must be present for every reachable state.
        self.access: Dict[int, List[int]] = {}
        #: ``incoming[s]`` -- the edges whose current target is ``s``, so that a
        #: split can re-open exactly those.
        self.incoming: Dict[int, Set[Tuple[int, int]]] = {
            s: set() for s in range(num_states)
        }
        self.worklist: "deque[Tuple[int, int]]" = deque()

    # -- edges --------------------------------------------------------------

    def target(self, state: int, c: int) -> Optional[int]:
        return self.transitions[state].get(c)

    def has_edge(self, state: int, c: int) -> bool:
        return c in self.transitions[state]

    def witness(self, state: int, c: int) -> Optional[List[int]]:
        return self.witnesses.get((state, c))

    def set_edge(self, state: int, c: int, target: int, witness) -> None:
        previous = self.transitions[state].get(c)
        if previous is not None:
            self.incoming[previous].discard((state, c))
        self.transitions[state][c] = target
        self.witnesses[state, c] = list(witness)
        self.incoming[target].add((state, c))

    def clear_edge(self, state: int, c: int) -> None:
        target = self.transitions[state].pop(c, None)
        self.witnesses.pop((state, c), None)
        if target is not None:
            self.incoming[target].discard((state, c))

    # -- the queue ----------------------------------------------------------

    def reopen(self, state: int, c: int) -> None:
        """Re-queue ``(state, c)``; :meth:`drain` dedups against edges that have
        since been resolved."""
        self.worklist.append((state, c))

    def open_every_edge(self, states) -> None:
        for state in states:
            for c in range(self.alphabet_size):
                self.reopen(state, c)

    def pending_probes(self) -> List[List[int]]:
        """``access[s] + [c]`` for every queued edge still needing resolution --
        the strings the next :meth:`drain` will sift, so a caller can warm them in
        one batch.  The queue only grows outside a drain, so one pass covers it."""
        return [
            list(self.access[s]) + [c]
            for s, c in self.worklist
            if s in self.access and not self.has_edge(s, c)
        ]

    def drain(self, resolve) -> int:
        """Resolve queued edges via ``resolve(state, symbol)`` until the
        hypothesis is closed.  Returns the number of edges resolved.

        A split reuses the old id for its True branch, so every id below the state
        count is always live -- no staleness check is needed, and the only dedup is
        skipping edges already resolved."""
        resolved = 0
        while self.worklist:
            state, c = self.worklist.popleft()
            if self.has_edge(state, c):
                continue
            resolve(state, c)
            resolved += 1
        return resolved

    # -- splitting ----------------------------------------------------------

    def split_state(self, state: int, new_state: int) -> None:
        """Account for ``state`` having bifurcated into ``state`` and
        ``new_state``.

        Only edges *incident* to the old leaf become ambiguous: its outgoing edges
        vanish (the source is now two states), and the edges pointing at it must be
        re-classified into one of the two.  Both sets are dropped and re-queued,
        along with every outgoing edge of the two halves."""
        self.transitions[new_state] = {}
        self.incoming[new_state] = set()
        for c in list(self.transitions[state]):
            self.clear_edge(state, c)
            self.reopen(state, c)
        for src, c in list(self.incoming[state]):
            self.clear_edge(src, c)
            self.reopen(src, c)
        self.incoming[state] = set()
        self.open_every_edge((state, new_state))

    # -- export -------------------------------------------------------------

    def totalise(self, states, decisive_target):
        """A total copy of ``delta``.  An edge the worklist left open is filled
        from ``decisive_target(state, symbol)``; where that fails too the edge
        self-loops and is reported in the second return value.

        Does not mutate ``transitions`` -- unresolved edges stay open so a later
        round can still close them."""
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
