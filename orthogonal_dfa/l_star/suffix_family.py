"""The suffix family a round classifies against.

A noisy oracle cannot classify a string with one query, so direct-L* averages
membership over a *family* of distinguishing suffixes and only answers when the
mean lands decisively past a threshold.  This owns that family: the suffixes
themselves, the memo of the means computed from them, and the ASSIGN/TEST
partition the split test needs.

The partition is the reason the halves live here rather than with the split
test.  Grouping a member and scoring the resulting split must not read the same
suffixes -- otherwise noise that pushed a member onto one side also shows up as
evidence for that side, and a leaf could be split by its own noise.  Alternating
indices rather than halving keeps both halves representative of the family
whatever order ``sample_suffix_family`` returned it in.
"""

import random
from typing import Dict, List, Optional, Tuple

from .statistics import binomial_side_of_boundary

#: Family sizes at which the sequential :meth:`SuffixFamily.is_accept` test looks
#: for a decision, smallest first.  A prefix far from the threshold settles in the
#: first block; only genuine boundary prefixes walk the whole family.
_SIFT_BLOCK = 16

#: Seed for the content-independent order the sequential test reveals suffixes in.
#: Fixed, and independent of the pipeline rng, so it neither perturbs downstream
#: sampling nor correlates with any structure in the family's own ordering -- each
#: block is then a representative sample of the whole family.
_SIFT_ORDER_SEED = 0

#: Chance the sequential test stops early on the wrong side of the boundary.  Set
#: small: an early stop trades the exact full-family mean for a decision from a
#: prefix of it, so it must be confident the full family would agree.
_SIFT_ALPHA = 1e-4

_MISSING = object()


class SuffixFamily:
    """See the module docstring."""

    def __init__(self, pst, vs: List[int]):
        self.pst = pst
        self.vs = list(vs)
        self.assign_idx = list(range(0, len(self.vs), 2))
        self.test_idx = list(range(1, len(self.vs), 2))
        # The order the sequential test reveals suffixes in -- a fixed shuffle, so
        # each block samples the whole family even when the family's own order
        # carries structure (screening admits suffixes in correlated batches).
        self._sift_order = random.Random(_SIFT_ORDER_SEED).sample(
            range(len(self.vs)), len(self.vs)
        )
        # Sift verdicts, memoized per (seq, midfix).  Per-round, because the
        # verdict depends on which family is in play; the underlying cells live in
        # the table, which persists across rounds.
        self._verdicts: Dict[Tuple[tuple, tuple], Optional[bool]] = {}

    def bits(self, base) -> List[int]:
        """Membership of ``base`` under each family suffix.

        A sift base is not a pool prefix -- a fresh string is touched at every
        tree node -- so this goes through the off-grid ``sift_cache`` rather than
        the table's grid.  Misses are issued as one batched call, so a batching
        oracle evaluates the whole family for a node in a single forward pass."""
        table = self.pst.table
        return self.pst.sift_cache.membership(
            [list(base) + table.suffix(v) for v in self.vs]
        )

    def prefill(self, bases) -> None:
        """Observe the whole family for every base at once, so a population costs
        one oracle call rather than one per member."""
        table = self.pst.table
        self.pst.sift_cache.membership(
            [list(b) + table.suffix(v) for b in bases for v in self.vs]
        )

    def warm_sift(self, bases) -> None:
        """Observe just the first sequential block for every base at once, so a
        whole tree level's easy sifts settle in one batched call.  :meth:`is_accept`
        draws deeper blocks per-base only for the prefixes a first block cannot
        resolve, so warming the full family here would be wasted on most of them."""
        upto = min(_SIFT_BLOCK, len(self.vs))
        self.pst.sift_cache.membership(
            [list(b) + s for b in bases for s in self._sift_suffixes(0, upto)]
        )

    def _sift_suffixes(self, lo: int, hi: int) -> List[list]:
        """The suffix strings at positions ``lo:hi`` of the sequential sift order."""
        table = self.pst.table
        return [table.suffix(self.vs[i]) for i in self._sift_order[lo:hi]]

    def is_accept(self, seq, midfix) -> Optional[bool]:
        """Confidently classify ``seq`` at ``midfix``: ``True`` / ``False`` when
        the family accept-rate lands past ``accept_thresh`` / ``reject_thresh``,
        and ``None`` in the indecisive band between them.  That band is what keeps
        a single leaf from being split twice on the same noise.

        The rate is estimated sequentially: family suffixes are drawn a block at a
        time and the test stops as soon as a binomial test is confident the full
        family would land past a threshold, spending the whole family only on the
        boundary prefixes that genuinely need it.  The full-family mean is used
        exactly once the family is exhausted, so an exhausted verdict matches a
        plain full-family decision."""
        key = (tuple(seq), tuple(midfix))
        cached = self._verdicts.get(key, _MISSING)
        if cached is not _MISSING:
            return cached
        verdict = self._sequential_decide(list(seq) + list(midfix))
        self._verdicts[key] = verdict
        return verdict

    def _sequential_decide(self, base) -> Optional[bool]:
        n = len(self.vs)
        accepts = 0
        drawn = 0
        upto = _SIFT_BLOCK
        while True:
            upto = min(upto, n)
            block = [list(base) + s for s in self._sift_suffixes(drawn, upto)]
            accepts += sum(self.pst.sift_cache.membership(block))
            drawn = upto
            if upto >= n:
                return self._decide(accepts / n, margin=0.0)
            # Confident the full-family rate is past a threshold?  Sampling the
            # family without replacement is tighter than the binomial models, so
            # this only ever stops later than strictly necessary, never sooner.
            if binomial_side_of_boundary(
                accepts, upto, self.pst.accept_thresh, failure_prob=_SIFT_ALPHA
            ):
                return True
            if (
                binomial_side_of_boundary(
                    accepts, upto, self.pst.reject_thresh, failure_prob=_SIFT_ALPHA
                )
                is False
            ):
                return False
            upto *= 2

    def _decide(self, value: float, *, margin: float) -> Optional[bool]:
        if value >= self.pst.accept_thresh + margin:
            return True
        if value < self.pst.reject_thresh - margin:
            return False
        return None

    def votes(self, seq, midfix) -> List[int]:
        """Per-suffix accept bits, so the ASSIGN and TEST halves can be summed
        separately for the split Bayes factor."""
        return self.bits(list(seq) + list(midfix))

    def assign_side(self, votes) -> Optional[bool]:
        """Which side of the distinguisher these votes fall on, judged on the
        ASSIGN half only -- so the TEST half stays independent of the grouping.
        ``None`` if indecisive there; such a member contributes no evidence."""
        mean = sum(votes[i] for i in self.assign_idx) / len(self.assign_idx)
        if mean >= self.pst.accept_thresh:
            return True
        if mean < self.pst.reject_thresh:
            return False
        return None
