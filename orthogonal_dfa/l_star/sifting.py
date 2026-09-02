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
        """Route ``seq`` to a leaf.  Uses the decisive :meth:`SuffixFamily.side`,
        so every node descends on the side of the ``decision_boundary`` its mean
        falls on and a leaf is always reached: the result is ``(state, None)``.

        Routing has to place a string somewhere; abstaining in the confident band
        (as :meth:`is_accept` does) only leaves the caller to self-loop the edge or
        drop the string, which strands and misroutes it.  The confident band is
        kept where it belongs -- the FNR gate and the split-evidence verdict -- not
        in the routing.  The ``None`` second element is retained for callers that
        harvest a boundary string, which now never fires."""
        return self.tree.sift(seq, self.family.side)

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

        This only *proposes* a distinguisher; whether the split fires is decided
        by the population evidence (:class:`SplitEvidence`), so proposing on the
        decisive :meth:`SuffixFamily.side` -- rather than requiring both members to
        clear the confident band -- costs nothing but lets a distinguisher that
        separates two borderline states still be found; the split test remains the
        gate on whether it is real."""
        return self.tree.first_disagreement(s, sprime, self.family.side, prefix)
