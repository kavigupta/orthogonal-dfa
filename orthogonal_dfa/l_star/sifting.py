"""Classifying strings against the current discrimination tree.

The tree knows which midfix cuts where; the family knows how to answer a midfix.
Putting them together is what "sift" means, and both the probe loop and the edge
resolver need it, so it lives here rather than in either of them.
"""

from collections import namedtuple
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

    def disagreement(self, s, sprime, prefix):
        """A midfix separating ``s`` and ``sprime``, or the boundary string of the
        read that stopped the search (see :meth:`MidfixTree.first_disagreement`).

        This only *proposes* a distinguisher; whether the split fires is decided
        by the population evidence, so the pair need only clear the ordinary
        decisive band, not a wide split margin."""
        return self.tree.first_disagreement(s, sprime, self.family.is_accept, prefix)


#: How reading a probe against a hypothesis ends (see ``read``).
AGREE = "agree"
START_UNDECIDED = "start undecided"
END_UNDECIDED = "end undecided"
PAIR = "pair"
EDGE = "edge"
TRIPLE = "triple"
UNLEARNED_EDGE = "unlearned edge"

#: ``at``: the length of the prefix the outcome is at (the read cut short, the
#: edge's head, a triple's middle, a pair's first, or before an unlearned edge);
#: ``string``: an undecided read's boundary string, both a pair's, or an
#: unlearned edge's member; ``state``: the walk's state before an edge, or the
#: one an unlearned edge's member reaches.
Outcome = namedtuple("Outcome", "kind at string state")


def read(probe, sift, transitions, k):
    """The outcome of walking ``probe`` along the learned ``transitions`` from
    where ``sift`` places its first ``k`` symbols, and sifting it whole.

    At an unlearned edge the prefixes either side of it are sifted.  Where the
    walk and the sift disagree, the search between them narrows on decided reads
    only, and ends at an edge (agree, disagree), a triple (agree, undecided,
    disagree) or a pair of adjacent undecided reads."""
    anchor, boundary = sift(probe[:k])
    if anchor is None:
        return Outcome(START_UNDECIDED, k, boundary, None)
    states = [None] * k + [anchor]
    for j in range(k, len(probe)):
        target = transitions[states[-1]].get(probe[j])
        if target is None:
            return _unlearned(probe, sift, states, k)
        states.append(target)
    end, boundary = sift(probe)
    if end is None:
        return Outcome(END_UNDECIDED, len(probe), boundary, None)
    if end == states[-1]:
        return Outcome(AGREE, len(probe), None, None)
    return _search(probe, sift, states, k)


def _unlearned(probe, sift, states, k):
    j = len(states) - 1
    for at in (j + 1, j):
        leaf, boundary = sift(probe[:at])
        if leaf is None:
            return Outcome(END_UNDECIDED, at, boundary, None)
    if leaf == states[j]:
        return Outcome(UNLEARNED_EDGE, j, probe[:j], leaf)
    return _search(probe, sift, states, k)


def _search(probe, sift, states, lo):
    """Where the walk ``states`` and ``sift`` part between ``lo``, where they
    agree, and the walk's end, where they disagree."""
    hi = len(states) - 1

    def agrees(p):
        if p in (lo, hi):
            return p == lo
        leaf = sift(probe[:p])[0]
        return None if leaf is None else leaf == states[p]

    while hi - lo > 1:
        mid = (lo + hi) // 2
        side = agrees(mid)
        if side is not None:
            lo, hi = (mid, hi) if side else (lo, mid)
            continue
        left = agrees(mid - 1)
        if left is None:
            return _pair(probe, sift, mid - 1)
        right = agrees(mid + 1)
        if right is None:
            return _pair(probe, sift, mid)
        if left and not right:
            return Outcome(TRIPLE, mid, sift(probe[:mid])[1], None)
        lo, hi = (lo, mid - 1) if not left else (mid + 1, hi)
    return Outcome(EDGE, hi, None, states[hi - 1])


def _pair(probe, sift, at):
    return Outcome(PAIR, at, (sift(probe[:at])[1], sift(probe[: at + 1])[1]), None)
