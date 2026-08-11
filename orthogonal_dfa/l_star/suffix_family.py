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

from typing import Dict, List, Optional, Tuple


class SuffixFamily:
    """See the module docstring."""

    def __init__(self, pst, vs: List[int]):
        self.pst = pst
        self.vs = list(vs)
        self.assign_idx = list(range(0, len(self.vs), 2))
        self.test_idx = list(range(1, len(self.vs), 2))
        # Family means, memoized per (seq, midfix).  Per-round, because the mean
        # depends on which family is in play; the underlying cells live in the
        # table, which persists across rounds.
        self._means: Dict[Tuple[tuple, tuple], float] = {}

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

    def mean(self, seq, midfix) -> float:
        """Mean family membership of ``seq`` under the distinguishers
        ``midfix + v``."""
        key = (tuple(seq), tuple(midfix))
        cached = self._means.get(key)
        if cached is not None:
            return cached
        value = sum(self.bits(list(seq) + list(midfix))) / len(self.vs)
        self._means[key] = value
        return value

    def is_accept(self, seq, midfix) -> Optional[bool]:
        """Confidently classify ``seq`` at ``midfix``: ``True`` / ``False`` when
        the family mean lands past ``accept_thresh`` / ``reject_thresh``, and
        ``None`` in the indecisive band between them.  That band is what keeps a
        single leaf from being split twice on the same noise."""
        return self._decide(self.mean(seq, midfix), margin=0.0)

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
