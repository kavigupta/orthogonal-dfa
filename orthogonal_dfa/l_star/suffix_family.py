"""The suffix family a classification round reads against.

A noisy oracle cannot classify a string with one query, so the learner averages
membership over a *family* of distinguishing suffixes and only answers when the
mean lands decisively past a threshold. This owns that family: the suffix rows
and the memo of the means computed from them.
"""

from typing import Dict, List, Optional, Tuple


class SuffixFamily:
    """The round's suffixes ``vs`` (rows into ``pst.table``), and confident
    classification of a string against a midfix node through their mean."""

    def __init__(self, pst, vs: List[int]):
        self.pst = pst
        self.vs = list(vs)
        # ASSIGN/TEST halves for the split test: grouping a member and scoring the
        # resulting split must read disjoint suffixes, or noise that pushed a
        # member onto one side shows up as evidence for that side and a leaf splits
        # on its own noise. Alternating keeps both halves representative.
        self.assign_idx = list(range(0, len(self.vs), 2))
        self.test_idx = list(range(1, len(self.vs), 2))
        self._means: Dict[Tuple[tuple, tuple], float] = {}

    def bits(self, base) -> List[int]:
        """Membership of ``base`` under each family suffix, through the table's
        shared memo so cells the mask already holds cost no new query."""
        table = self.pst.table
        return table.memo.membership_queries(
            [list(base) + table.suffix(v) for v in self.vs]
        )

    def prefill(self, bases) -> None:
        """Observe the whole family for every base at once, so a population costs
        one oracle call rather than one per member."""
        table = self.pst.table
        table.memo.membership_queries(
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
        ``None`` in the indecisive band between them."""
        mean = self.mean(seq, midfix)
        if mean >= self.pst.accept_thresh:
            return True
        if mean < self.pst.reject_thresh:
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
