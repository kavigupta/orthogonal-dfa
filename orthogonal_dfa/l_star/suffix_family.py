"""The suffix family a classification round reads against.

A noisy oracle cannot classify a string with one query, so the learner averages
membership over a *family* of distinguishing suffixes and only answers when the
mean lands decisively past a threshold.
"""

from typing import Dict, List, Optional


class SuffixFamily:
    """The round's suffixes ``vs`` (rows into ``pst.table``)."""

    def __init__(self, pst, vs: List[int]):
        self.pst = pst
        self.vs = list(vs)
        # A later round moves pst's boundary; this round's tree was cut at these.
        self.accept_thresh = pst.accept_thresh
        self.reject_thresh = pst.reject_thresh
        self._means: Dict[bytes, float] = {}

    def read_many(self, bases) -> None:
        """Every base's mean, in one call to the memo."""
        new = [base for base in dict.fromkeys(bases) if base not in self._means]
        if not new:
            return
        table = self.pst.table
        bits = table.memo.membership_queries(
            [base + table.suffix(v) for base in new for v in self.vs]
        )
        size = len(self.vs)
        for i, base in enumerate(new):
            self._means[base] = sum(bits[i * size : (i + 1) * size]) / size

    def is_accept(self, seq, midfix) -> Optional[bool]:
        """``None`` in the band between the thresholds."""
        base = seq + midfix
        self.read_many([base])
        mean = self._means[base]
        if mean >= self.accept_thresh:
            return True
        if mean < self.reject_thresh:
            return False
        return None
