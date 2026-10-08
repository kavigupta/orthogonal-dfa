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

    def halfway(self, seq) -> int:
        """Where the gate's reading sends ``seq``: past each node the cut cannot
        place it at, the side of the middle of the band."""
        return self.tree.route_halfway(
            seq, self.family.is_accept, self.family.middle_side
        )

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


#: How a search between an agreeing and a disagreeing prefix ends: at an edge
#: (agree, disagree), at an undecided read between them (agree, undecided,
#: disagree), or at two adjacent undecided reads.
EDGE, TRIPLE, PAIR = "edge", "triple", "pair"


def bracket(probe, states, sift, lo, hi):
    """``(kind, at)``: where the walk ``states`` and ``sift`` part between ``lo``,
    where they agree, and ``hi``, where they disagree, reading only decided
    prefixes to narrow.  ``at`` is the edge's head, the undecided middle of a
    triple, or the first of a pair."""

    def agrees(p):
        if p in (lo, hi):
            return p == lo
        leaf = sift(probe[:p])
        return None if leaf is None else leaf == states[p]

    while hi - lo > 1:
        mid = (lo + hi) // 2
        side = agrees(mid)
        if side is not None:
            lo, hi = (mid, hi) if side else (lo, mid)
            continue
        left = agrees(mid - 1)
        if left is None:
            return PAIR, mid - 1
        right = agrees(mid + 1)
        if right is None:
            return PAIR, mid
        if left and not right:
            return TRIPLE, mid
        lo, hi = (lo, mid - 1) if not left else (mid + 1, hi)
    return EDGE, hi


#: Where a walk stopped, at ``probe[:at]``, what it leaves there, and whether
#: that is a string the cut could not place.  A blocked start or edge leaves the
#: prefix the cut could not place; at an open edge whose next prefix the cut
#: places, the prefix before it, a member of the edge's state.  A blocked sift of
#: the whole probe leaves its boundary string.
Block = namedtuple("Block", "at found undecided")


def walk(probe, sift, transitions, k):
    """``(states, block, end)``.  ``states[i]`` is the state the learned
    ``transitions`` reach after ``probe[:i]`` from where ``sift`` places
    ``probe[:k]``, ``None`` below ``k``, as far as the walk got; ``block`` is what
    stopped it, or ``None``; ``end`` is where ``sift`` places the prefix the walk
    ended on: the whole probe where nothing blocked, or the prefix before an open
    edge that sifts to another state than the walk's.
    ``sift`` answers as :meth:`Sifter.sift_and_boundary` does."""
    states = [None] * k
    anchor = sift(probe[:k])[0]
    if anchor is None:
        return states, Block(k, probe[:k], True), None
    states.append(anchor)
    for j in range(k, len(probe)):
        target = transitions[states[-1]].get(probe[j])
        if target is None:
            return (states, *_edge_block(probe, sift, states[-1], j))
        states.append(target)
    end, boundary = sift(probe)
    if end is None:
        return states, Block(len(probe), boundary, True), None
    return states, None, end


def _edge_block(probe, sift, state, j):
    if sift(probe[: j + 1])[0] is None:
        return Block(j + 1, probe[: j + 1], True), None
    before = sift(probe[:j])[0]
    if before is None:
        return Block(j + 1, probe[:j], True), None
    if before == state:
        return Block(j + 1, probe[:j], False), None
    return None, before
