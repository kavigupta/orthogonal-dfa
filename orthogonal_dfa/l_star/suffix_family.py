"""The suffix family a classification round reads against.

A noisy oracle cannot classify a string with one query, so the learner averages
membership over a *family* of distinguishing suffixes and only answers when the
mean lands decisively past a threshold. This owns that family: the suffix rows
and the memo of the means computed from them.
"""

from typing import Dict, List, Optional, Tuple

#: Suffixes read at a time while a node's verdict is still open.
READ_BLOCK = 8


class SuffixFamily:
    """The round's suffixes ``vs`` (rows into ``pst.table``), and confident
    classification of a string against a midfix node through their mean."""

    def __init__(self, pst, vs: List[int]):
        self.pst = pst
        self.vs = list(vs)
        # A later round moves pst's boundary; this round's tree was cut at these.
        self.accept_thresh = pst.accept_thresh
        self.reject_thresh = pst.reject_thresh
        #: The middle of the band, where the gate reads.
        self.middle = (self.accept_thresh + self.reject_thresh) / 2
        # train/test halves for the split test
        self.train_idx = list(range(0, len(self.vs), 2))
        self.test_idx = list(range(1, len(self.vs), 2))
        # keyed by seq + midfix, which is all a mean depends on
        self._means: Dict[bytes, float] = {}
        #: Per base, its cut verdict and middle side, where those were fixed
        #: before every suffix was read.
        self._sides: Dict[bytes, Tuple[Optional[bool], bool]] = {}

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
        return seq + midfix in self._means or seq + midfix in self._sides

    def is_accept(self, seq, midfix) -> Optional[bool]:
        """Confidently classify ``seq`` at ``midfix``: ``True`` / ``False`` when
        the family mean lands past ``accept_thresh`` / ``reject_thresh``, and
        ``None`` in the indecisive band between them."""
        return self._sides_of(seq + midfix)[0]

    def middle_side(self, seq, midfix) -> bool:
        """Whether the family mean lands above the middle of the band; exactly on
        it reads as reject."""
        return self._sides_of(seq + midfix)[1]

    def _cut(self, mean) -> Optional[bool]:
        if mean >= self.accept_thresh:
            return True
        if mean < self.reject_thresh:
            return False
        return None

    def _sides_of(self, base) -> Tuple[Optional[bool], bool]:
        """``base``'s cut verdict and middle side, reading its suffixes
        ``READ_BLOCK`` at a time only until no rest of them could move
        either."""
        if base in self._means:
            mean = self._means[base]
            return self._cut(mean), mean > self.middle
        if base in self._sides:
            return self._sides[base]
        table, n = self.pst.table, len(self.vs)
        ones = read = 0
        while True:
            block = self.vs[read : read + READ_BLOCK]
            ones += sum(
                table.memo.membership_queries([base + table.suffix(v) for v in block])
            )
            read += len(block)
            low, high = ones / n, (ones + n - read) / n
            if self._cut(low) == self._cut(high) and (low > self.middle) == (
                high > self.middle
            ):
                break
        if read == n:
            self._means[base] = low
        sides = self._sides[base] = (self._cut(low), low > self.middle)
        return sides

    def votes(self, seq, midfix) -> List[int]:
        """Per-suffix accept bits"""
        bits = self.bits(seq + midfix)
        self._means.setdefault(seq + midfix, sum(bits) / len(self.vs))
        return bits

    def train_side(self, votes) -> Optional[bool]:
        """
        Which side of the distinguisher the votes fall on (on the training half only).
        """
        mean = sum(votes[i] for i in self.train_idx) / len(self.train_idx)
        if mean >= self.accept_thresh:
            return True
        if mean < self.reject_thresh:
            return False
        return None
