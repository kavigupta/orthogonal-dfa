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
        """Route ``seq`` to a leaf: ``(state, None)``, or ``(None, boundary)`` when
        a node lands in the confident band and cannot place it.  The banded sift
        runs first so the boundary is harvested for the next round; a caller that
        must commit anyway falls back to :meth:`sift_decisive`."""
        return self.tree.sift(seq, self.family.is_accept)

    def sift_decisive(self, seq) -> Optional[int]:
        """Route ``seq`` to a leaf with the zero-band decisive classifier
        (:meth:`SuffixFamily.side`), so every node descends and a leaf is always
        reached.  This is the routing *fallback* for when :meth:`sift_and_boundary`
        abstains: the string still has to go somewhere, and a self-loop is no
        better a guess than the side of the ``decision_boundary`` its mean falls
        on.  The banded sift runs first, so the boundary is still harvested; only a
        string the confident sift could not place is forced to decide here."""
        return self.tree.sift(seq, self.family.side)[0]

    def sift(self, seq) -> Optional[int]:
        """The leaf ``seq`` sifts to, or ``None`` -- the boundary discarded.  A
        non-harvesting classifier, used by the diagnostics renderer (which must not
        pollute ``indecisive`` while drawing)."""
        return self.sift_and_boundary(seq)[0]

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

        The confident classifier runs first; if it finds nothing -- which includes
        a needed classification landing in the band -- it falls back to the
        decisive :meth:`SuffixFamily.side`, so a distinguisher separating two
        borderline states can still be *proposed*.  It only proposes;
        :class:`SplitEvidence` remains the gate on whether the split is real."""
        found = self.tree.first_disagreement(s, sprime, self.family.is_accept, prefix)
        if found is not None:
            return found
        return self.tree.first_disagreement(s, sprime, self.family.side, prefix)
