"""The suffix family a classification round reads against.

A noisy oracle cannot classify a string with one query, so the learner averages
membership over a *family* of distinguishing suffixes and only answers when the
mean lands decisively past a threshold. This owns that family: the suffix rows
and the memo of the means computed from them.
"""

from typing import Dict, List, Optional


class SuffixFamily:
    """The round's suffixes ``vs`` (rows into ``pst.table``), and confident
    classification of a string against a midfix node through their mean."""

    def __init__(self, pst, vs: List[int]):
        self.pst = pst
        self.vs = list(vs)
        # A later round moves pst's boundary; this round's tree was cut at these.
        self.accept_thresh = pst.accept_thresh
        self.reject_thresh = pst.reject_thresh
        # keyed by seq + midfix, which is all a mean depends on
        self._means: Dict[bytes, float] = {}

    def bits(self, base) -> List[int]:
        """Membership of ``base`` under each family suffix, through the table's
        shared memo so cells the mask already holds cost no new query."""
        table = self.pst.table
        return table.memo.membership_queries([base + table.suffix(v) for v in self.vs])

    def mean(self, seq, midfix) -> float:
        """Mean family membership of ``seq`` under the distinguishers
        ``midfix + v``."""
        base = seq + midfix
        cached = self._means.get(base)
        if cached is not None:
            return cached
        value = sum(self.bits(base)) / len(self.vs)
        self._means[base] = value
        return value

    def is_accept(self, seq, midfix) -> Optional[bool]:
        """Confidently classify ``seq`` at ``midfix``: ``True`` / ``False`` when
        the family mean lands past ``accept_thresh`` / ``reject_thresh``, and
        ``None`` in the indecisive band between them."""
        mean = self.mean(seq, midfix)
        if mean >= self.accept_thresh:
            return True
        if mean < self.reject_thresh:
            return False
        return None
