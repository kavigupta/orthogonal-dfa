"""Classifying strings against the current discrimination tree.

The tree knows which midfix cuts where; the family knows how to answer a midfix.
Putting them together is what "sift" means, and both the probe loop and the edge
resolver need it, so it lives here rather than in either of them.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

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

    def halfway(self, seq) -> Tuple[int, List[bytes]]:
        """Where the gate's reading sends ``seq``: past each node the cut cannot
        place it at, the side of the middle of the band; with the strings read
        at those nodes.  Its reads are not the pass's, so ``reads`` leaves them
        out."""
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

    def disagreement(
        self, s, sprime, prefix
    ) -> Tuple[Optional[bytes], Optional[bytes]]:
        """A midfix separating ``s`` and ``sprime``, or the string the search
        could not place (see :meth:`MidfixTree.first_disagreement`).

        This only *proposes* a distinguisher; whether the split fires is decided
        by the population evidence, so the pair need only clear the ordinary
        decisive band, not a wide split margin."""
        return self.tree.first_disagreement(s, sprime, self.family.is_accept, prefix)


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


#: Where a walk from the start length can stop: the sift of the probe's start,
#: an edge the hypothesis has not learned, or the sift of the whole probe.
ANCHOR, EDGE, END = "anchor", "edge", "end"


@dataclass(frozen=True)
class Block:
    """Where a probe was blocked, at ``probe[:at]``, what it leaves there, and
    whether that is a string the cut could not place.

    A blocked walk leaves the prefix it could not place.  At an open edge whose
    next prefix the cut places, that is the prefix before it: a member where it
    sifts to the edge's state, and nothing where it sifts elsewhere.  A blocked
    sift of the whole probe leaves its boundary string."""

    kind: str
    at: int
    found: Optional[bytes]
    undecided: bool


def walk_from(probe, sift, transitions, k):
    """``(states, block)``: ``states[i]`` is the state after ``probe[:i]`` as the
    learned ``transitions`` reach it from where ``sift`` places ``probe[:k]``,
    ``None`` below ``k``, and as far as the walk got; ``block`` is what stopped it,
    or ``None``.  ``sift`` answers as :meth:`Sifter.sift_and_boundary` does."""
    anchor, _ = sift(probe[:k])
    if anchor is None:
        return None, Block(ANCHOR, k, probe[:k], True)
    states = [None] * k + [anchor]
    for j in range(k, len(probe)):
        target = transitions[states[-1]].get(probe[j])
        if target is None:
            return states, _edge_block(probe, sift, states[j], j)
        states.append(target)
    return states, None


def _edge_block(probe, sift, state, j):
    if sift(probe[: j + 1])[0] is None:
        return Block(EDGE, j + 1, probe[: j + 1], True)
    before, _ = sift(probe[:j])
    if before is None:
        return Block(EDGE, j + 1, probe[:j], True)
    return Block(EDGE, j + 1, probe[:j] if before == state else None, False)


def read_from(probe, sift, transitions, k):
    """``(states, block, disagrees)``: the walk from ``k`` (see :func:`walk_from`)
    and, where it went through, the cut's sift of the whole probe, which blocks it
    where the cut cannot place it and otherwise says whether the walk's end
    disagrees with it."""
    states, block = walk_from(probe, sift, transitions, k)
    if block is not None:
        return states, block, False
    end, boundary = sift(probe)
    if end is None:
        return states, Block(END, len(probe), boundary, True), False
    return states, None, end != states[-1]


def check_from(probe, sift, middle, transitions, k):
    """``(states, block, fd)``: :func:`read_from`, and, where the walk's end and
    the sift disagree, ``fd``, the first index where they part (see
    :func:`first_disagreeing_edge`).  The search places a prefix the cut cannot
    by ``middle``."""
    states, block, disagrees = read_from(probe, sift, transitions, k)
    if not disagrees:
        return states, block, None

    def settle(seq):
        leaf, _ = sift(seq)
        return middle(seq) if leaf is None else leaf

    return states, None, first_disagreeing_edge(probe, states, settle, k, len(probe))
