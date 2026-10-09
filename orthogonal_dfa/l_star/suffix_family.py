"""The suffix family a classification round reads against.

A noisy oracle cannot classify a string with one query, so the learner averages
membership over a *family* of distinguishing suffixes and only answers when the
mean lands decisively past a threshold. This owns that family: the suffix rows
and the memo of the means computed from them.
"""

from typing import Dict, List, Optional


class SuffixFamily:
    """The round's suffixes ``vs`` (rows into ``pst.table``), and confident
    classification of a string against a midfix node through their mean;
    ``held_out``, rows only the split test reads."""

    def __init__(self, pst, vs: List[int], held_out: List[int]):
        self.pst = pst
        self.vs = list(vs)
        self.held_out = list(held_out)
        # A later round moves pst's boundary; this round's tree was cut at these.
        self.accept_thresh = pst.accept_thresh
        self.reject_thresh = pst.reject_thresh
        #: The middle of the band, where the gate reads.
        self.middle = (self.accept_thresh + self.reject_thresh) / 2
        # The half a split test groups members on.
        self.train_idx = list(range(0, len(self.vs), 2))
        # keyed by seq + midfix, which is all a mean depends on
        self._means: Dict[bytes, float] = {}

    def bits(self, base) -> List[int]:
        """Membership of ``base`` under each family suffix, through the table's
        shared memo so cells the mask already holds cost no new query."""
        table = self.pst.table
        return table.memo.membership_queries([base + table.suffix(v) for v in self.vs])

    def held_out_bits(self, strings) -> List[int]:
        """Membership of each of ``strings``, which are bases followed by
        held-out suffixes."""
        return self.pst.table.memo.membership_queries(strings)

    def held_out_strings(self, base) -> List[bytes]:
        return [base + self.pst.table.suffix(v) for v in self.held_out]

    def unread(self, strings) -> List[bytes]:
        """Those of ``strings`` no read in the run has asked."""
        known = self.pst.table.memo.known(strings) if strings else []
        return [s for s, k in zip(strings, known) if not k]

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
        return seq + midfix in self._means

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

    def middle_side(self, seq, midfix) -> bool:
        """Whether the family mean lands above the middle of the band; exactly on
        it reads as reject."""
        return self.mean(seq, midfix) > self.middle

    def votes(self, seq, midfix) -> List[int]:
        """Per-suffix accept bits"""
        bits = self.bits(seq + midfix)
        self._means.setdefault(seq + midfix, sum(bits) / len(self.vs))
        return bits

    def train_side(self, votes) -> Optional[bool]:
        """
        Which side of the distinguisher the votes fall on (on the training half
        only): accept past ``accept_thresh``, reject at or below
        ``reject_thresh``.
        """
        accepts = sum(votes[i] for i in self.train_idx)
        if accepts > self.accept_thresh * len(self.train_idx):
            return True
        if accepts <= self.reject_thresh * len(self.train_idx):
            return False
        return None
