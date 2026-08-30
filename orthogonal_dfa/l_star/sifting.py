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
        by the population evidence, so the pair need only clear the ordinary
        decisive band, not a wide split margin."""
        return self.tree.first_disagreement(s, sprime, self.family.is_accept, prefix)
