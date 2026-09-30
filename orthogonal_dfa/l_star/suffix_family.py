"""The suffix family a classification round reads against.

A noisy oracle cannot classify a string with one query, so the learner averages
membership over a *family* of distinguishing suffixes and only answers when the
mean lands decisively past a threshold. This owns that family: the suffix rows
and the memo of the means computed from them.
"""

from typing import Callable, Dict, List, Optional

#: Families' worth of reserve a read left in the band is read again over.
REREAD_FAMILIES = 3


class SuffixFamily:
    """The round's suffixes ``vs`` (rows into ``pst.table``), and confident
    classification of a string against a midfix node through their mean."""

    def __init__(self, pst, vs: List[int], more_suffixes: Callable[[], List[int]]):
        self.pst = pst
        self.vs = list(vs)
        # Drawn a family's worth at a time, only when a read needs it;
        # ``more_suffixes`` returns [] once none are left.
        self._more_suffixes = more_suffixes
        self.reserve: List[int] = []
        # train/test halves for the split test
        self.train_idx = list(range(0, len(self.vs), 2))
        self.test_idx = list(range(1, len(self.vs), 2))
        # keyed by seq + midfix, which is all a mean depends on
        self._means: Dict[bytes, float] = {}
        self._rereads: Dict[bytes, Optional[bool]] = {}

    def bits(self, base) -> List[int]:
        """Membership of ``base`` under each family suffix, through the table's
        shared memo so cells the mask already holds cost no new query."""
        table = self.pst.table
        return table.memo.membership_queries([base + table.suffix(v) for v in self.vs])

    def prefill(self, bases) -> None:
        """Observe the whole family for every base at once, so a population costs
        one oracle call rather than one per member."""
        table = self.pst.table
        table.memo.membership_queries(
            [b + table.suffix(v) for b in bases for v in self.vs]
        )

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

    def knows(self, seq, midfix) -> bool:
        """Whether ``is_accept`` can answer without a new query."""
        base = seq + midfix
        if base not in self._means:
            return False
        return self._side(self._means[base]) is not None or base in self._rereads

    def is_accept(self, seq, midfix) -> Optional[bool]:
        """Confidently classify ``seq`` at ``midfix``: ``True`` / ``False`` when
        the family mean lands past ``accept_thresh`` / ``reject_thresh``, and
        ``None`` in the indecisive band between them.

        A mean in the band is read again over the reserve, against the same
        thresholds."""
        side = self._side(self.mean(seq, midfix))
        if side is not None:
            return side
        base = seq + midfix
        if base not in self._rereads:
            self._rereads[base] = self._reread(base)
        return self._rereads[base]

    def _reread(self, base) -> Optional[bool]:
        table = self.pst.table
        bits = list(self.bits(base))
        size = len(self.vs)
        for look in range(REREAD_FAMILIES):
            if len(self.reserve) < (look + 1) * size:
                self.reserve += self._more_suffixes()
            more = self.reserve[look * size : (look + 1) * size]
            if not more:
                return None
            bits += list(
                table.memo.membership_queries([base + table.suffix(v) for v in more])
            )
            side = self._side(sum(bits) / len(bits))
            if side is not None:
                return side
        return None

    def _side(self, mean) -> Optional[bool]:
        if mean >= self.pst.accept_thresh:
            return True
        if mean < self.pst.reject_thresh:
            return False
        return None

    def votes(self, seq, midfix) -> List[int]:
        """Per-suffix accept bits"""
        bits = self.bits(seq + midfix)
        self._means.setdefault(seq + midfix, sum(bits) / len(self.vs))
        return bits

    def train_side(self, votes) -> Optional[bool]:
        """
        Which side of the distinguisher the votes fall on (on the training half only).
        """
        return self._side(sum(votes[i] for i in self.train_idx) / len(self.train_idx))
