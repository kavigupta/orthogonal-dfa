"""Classifying strings against the current discrimination tree.

The tree knows which midfix cuts where; the family knows how to answer a midfix.
Putting them together is what "sift" means, and both the probe loop and the edge
resolver need it, so it lives here rather than in either of them.
"""

from typing import Optional, Tuple

from .tripartite_search import EDGE, tripartite_search

#: Probes sifted per batched pass.
PROBE_BLOCK = 16


class Sifter:
    """Routes strings through ``tree``, classifying with ``family``."""

    def __init__(self, tree, family):
        self.tree = tree
        self.family = family
        #: Node reads every sift so far has made.
        self.reads = 0

    def sift_and_boundary(self, seq) -> Tuple[Optional[int], Optional[bytes]]:
        """Route ``seq`` to a leaf: ``(state, None)``, or ``(None, boundary)``
        when some node cannot place it."""

        def decide(s, midfix):
            self.reads += 1
            return self.family.is_accept(s, midfix)

        return self.tree.sift(seq, decide)

    def known_sift(self, seq) -> Optional[int]:
        """The leaf ``seq`` sifts to without a new query, or ``None`` when some
        node's read is not already memoized or is indecisive."""

        def decide(s, midfix):
            return (
                self.family.is_accept(s, midfix)
                if self.family.knows(s, midfix)
                else None
            )

        return self.tree.classify(seq, decide)

    def prefill(self, seqs) -> None:
        """Warm the cache for sifting all of ``seqs``, one batched call per tree
        level rather than one per node visited.

        Uses ``classify_many`` purely for its per-level walk: each level's
        ``decide`` first warms the whole level's family cells in one batched call,
        then reads them back (now cached) to descend.  The returned leaves are
        discarded -- only the warmed cache matters."""

        def warm(pairs):
            self.family.prefill([s + m for s, m in pairs])
            return [self.family.is_accept(s, m) for s, m in pairs]

        self.tree.classify_many(seqs, warm)

    def disagreement(self, s, sprime, prefix) -> Optional[bytes]:
        """A midfix separating ``s`` and ``sprime`` (see
        :meth:`MidfixTree.first_disagreement`), or ``None``.

        This only *proposes* a distinguisher; whether the split fires is decided
        by the population evidence, so the pair need only clear the ordinary
        decisive band, not a wide split margin."""
        return self.tree.first_disagreement(s, sprime, self.family.is_accept, prefix)


def anchored_walk(probe, sift, transitions, earliest):
    """Where ``sift`` first places a prefix of ``probe`` at least ``earliest``
    long, and what following ``transitions`` from there reaches.

    ``states[i]`` is the state after ``probe[:i]``, ``None`` below the anchor;
    ``(None, None)`` where no such prefix places.
    """
    start = earliest
    while start < len(probe):
        state = sift(probe[:start])
        if state is not None:
            break
        start += 1
    else:
        return None, None
    states = [None] * start + [state]
    for symbol in probe[start:]:
        state = transitions[state][symbol]
        states.append(state)
    return start, states


def first_disagreeing_edge(probe, states, sift, lo, hi):
    """The first index where the walk and a fresh sift diverge, or ``None``
    where a sift on the way comes out indecisive.

    Invariant: the sift agrees at ``lo`` and disagrees at ``hi``, so neither is
    sifted again.
    """

    def agrees(i):
        if i in (lo, hi):
            return i == lo
        landed = sift(probe[:i])
        return None if landed is None else landed == states[i]

    found = tripartite_search(agrees, lo, hi, recover=False)
    return found.position if found.kind == EDGE else None
