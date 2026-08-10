"""Classifying strings against the current discrimination tree.

The tree knows which midfix cuts where; the family knows how to answer a midfix.
Putting them together is what "sift" means, and both the probe loop and the edge
resolver need it, so it lives here rather than in either of them.
"""

from typing import List, Optional, Tuple

#: Strings a pool scan sifts per batched pass.
SCAN_BLOCK = 128


class Sifter:
    """Routes strings through ``tree``, classifying with ``family``."""

    def __init__(self, tree, family):
        self.tree = tree
        self.family = family

    def sift_and_boundary(self, seq) -> Tuple[Optional[int], Optional[tuple]]:
        """Route ``seq`` to a leaf: ``(state, None)``, or ``(None, boundary)``
        when some node cannot place it."""
        return self.tree.sift(seq, self.family.is_accept)

    def sift(self, seq) -> Optional[int]:
        leaf, _ = self.sift_and_boundary(seq)
        return leaf

    def prefill(self, seqs) -> None:
        """Warm the cache for sifting all of ``seqs``, one batched call per tree
        level rather than one per node visited.

        Uses ``classify_many`` purely for its per-level walk: each level's
        ``decide`` first warms the whole level's family cells in one batched call,
        then reads them back (now cached) to descend.  The returned leaves are
        discarded -- only the warmed cache matters."""

        def warm(pairs):
            self.family.prefill([list(s) + list(m) for s, m in pairs])
            return [self.family.is_accept(s, m) for s, m in pairs]

        self.tree.classify_many(seqs, warm)

    def disagreement(self, s, sprime, prefix) -> Optional[tuple]:
        """A midfix separating ``s`` and ``sprime`` (see
        :meth:`MidfixTree.first_disagreement`), or ``None``.

        This only *proposes* a distinguisher; whether the split fires is decided
        by the population evidence, so the pair need only clear the ordinary
        decisive band, not a wide split margin."""
        return self.tree.first_disagreement(s, sprime, self.family.is_accept, prefix)

    def leaves_of(
        self, prefixes, state: int, *, limit: Optional[int]
    ) -> List[List[int]]:
        """Those of ``prefixes`` that sift to ``state``, scanned a block at a time.

        One batched call per tree level per block, rather than one per prefix.  A
        block overshoots ``limit`` by at most its own size, and that work only
        warms the cache the next scan reads back."""
        out: List[List[int]] = []
        for i in range(0, len(prefixes), SCAN_BLOCK):
            chunk = prefixes[i : i + SCAN_BLOCK]
            self.prefill(chunk)
            for p in chunk:
                if self.sift(p) == state:
                    out.append(p)
                    if limit is not None and len(out) >= limit:
                        return out
        return out
