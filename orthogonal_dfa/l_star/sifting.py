"""Classifying strings against the current discrimination tree.

The tree knows which midfix cuts where; the family knows how to answer a midfix.
Putting them together is what "sift" means, and both the probe loop and the edge
resolver need it, so it lives here rather than in either of them.
"""

from typing import Optional, Tuple


class Sifter:
    """Routes strings through ``tree``, classifying with ``family``."""

    def __init__(self, tree, family):
        self.tree = tree
        self.family = family

    def sift_and_boundary(self, seq) -> Tuple[Optional[int], Optional[bytes]]:
        """Route ``seq`` to a leaf: ``(state, None)``, or ``(None, boundary)``
        when some node cannot place it."""
        return self.tree.sift(seq, self.family.is_accept)

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


def anchored_walk(probe, sift, transitions):
    """Where ``sift`` first places a prefix of ``probe``, and what following
    ``transitions`` from there reaches.

    ``states[i]`` is the state after ``probe[:i]``, ``None`` below the anchor;
    ``(None, None)`` where no prefix places at all.
    """
    start = 0
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

    Invariant: the sift agrees at ``lo`` and disagrees at ``hi``.
    """
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        landed = sift(probe[:mid])
        if landed is None:
            return None
        lo, hi = (mid, hi) if landed == states[mid] else (lo, mid)
    return hi
