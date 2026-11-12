"""The suffix family a classification round reads against.

A noisy oracle cannot classify a string with one query, so the learner averages
membership over a *family* of distinguishing suffixes and only answers when the
mean lands decisively past a threshold.
"""

from typing import Dict, List, Optional

from .sequential_decide import sequential_decisions


class SuffixFamily:
    """The round's suffixes ``vs`` (rows into ``pst.table``)."""

    def __init__(self, pst, vs: List[int]):
        self.pst = pst
        self.vs = list(vs)
        # A later round moves pst's boundary; this round's tree was cut at these.
        self.accept_thresh = pst.accept_thresh
        self.reject_thresh = pst.reject_thresh
        self._sides: Dict[bytes, Optional[bool]] = {}

    def read_many(self, bases) -> None:
        """Every base's side, each read only as far as it takes to settle."""
        new = [base for base in dict.fromkeys(bases) if base not in self._sides]
        if not new:
            return
        table = self.pst.table
        sides = sequential_decisions(
            new,
            [table.suffix(v) for v in self.vs],
            table.memo.membership_queries,
            accept=self.accept_thresh,
            reject=self.reject_thresh,
        )
        self._sides.update(zip(new, sides))

    def is_accept(self, seq, midfix) -> Optional[bool]:
        """``None`` in the band between the thresholds."""
        base = seq + midfix
        self.read_many([base])
        return self._sides[base]
