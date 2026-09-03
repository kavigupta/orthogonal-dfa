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
        # train/test halves for the split test
        self.train_idx = list(range(0, len(self.vs), 2))
        self.test_idx = list(range(1, len(self.vs), 2))
        self._means: Dict[Tuple[bytes, bytes], float] = {}
        self._bits: Dict[bytes, bytes] = {}

    def bits(self, base) -> bytes:
        """Membership of ``base`` under each family suffix, indexed as ``vs``.

        Held per base rather than re-read: prefill has usually already observed
        the base, and rebuilding its row means concatenating and hashing a string
        per suffix to reach cells the memo already holds."""
        row = self._bits.get(base)
        if row is None:
            self._observe([base])
            row = self._bits[base]
        return row

    def prefill(self, bases) -> None:
        """Observe the whole family for every base at once, so a population costs
        one oracle call rather than one per member."""
        self._observe(list(bases))

    def _observe(self, keys) -> None:
        """Fill ``_bits`` for the bases among ``keys`` it does not already hold."""
        fresh = list(dict.fromkeys(k for k in keys if k not in self._bits))
        if not fresh:
            return
        table = self.pst.table
        suffixes = [table.suffix(v) for v in self.vs]
        answers = table.memo.membership_queries(
            [k + suffix for k in fresh for suffix in suffixes]
        )
        width = len(self.vs)
        for i, k in enumerate(fresh):
            self._bits[k] = bytes(answers[i * width : (i + 1) * width])

    def mean(self, seq, midfix) -> float:
        """Mean family membership of ``seq`` under the distinguishers
        ``midfix + v``."""
        key = (seq, midfix)
        cached = self._means.get(key)
        if cached is not None:
            return cached
        value = sum(self.bits(seq + midfix)) / len(self.vs)
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

    def votes(self, seq, midfix) -> bytes:
        """Per-suffix accept bits"""
        return self.bits(seq + midfix)

    def train_side(self, votes) -> Optional[bool]:
        """
        Which side of the distinguisher the votes fall on (on the training half only).
        """
        mean = sum(votes[i] for i in self.train_idx) / len(self.train_idx)
        if mean >= self.pst.accept_thresh:
            return True
        if mean < self.pst.reject_thresh:
            return False
        return None
