"""Edges a round reads badly, found where the boundary population would dilute them.

An edge is a source of prefixes extended by a letter.  A target state the family
reads badly but that is rare among reads hides in the boundary population; an
edge leading into it finds it at its share among the source's draws, so the edge
is judged on its own against the clean per-read rate the family is sized for:
promoted to a population of its own, rolled over into a chain the next round
tests afresh, or dropped.  A chain keeps only the draws whose extension the
round's tree could not place, so each round it is rolled over it is richer in the
states leading into the badly read one.
"""

from typing import List, Optional, Tuple

from .rejection_source import RejectionSource, SourceDry, proving_attempts
from .statistics import binomial_side_of_boundary

PROMOTE, ROLL_OVER, DROP = "promote", "roll over", "drop"

#: An edge undecided more than this many times the clean per-read rate is mostly
#: a badly read state: f > uHi / (uHi - 2 tau), which is 2 at the gap uHi = 4 tau.
EDGE_FACTOR = 3
#: Chance a round misjudges any of the edges it tests, split evenly over them.
EDGE_MISJUDGE = 0.01
#: Replays an edge gets before it is judged on its point estimate.
EDGE_MAX_PROBES = 300


def edge_verdict(undecided, reads, clean, *, failure_prob, final) -> Optional[str]:
    """Where an edge undecided on ``undecided`` replays over ``reads`` node reads
    stands against ``clean`` and EDGE_FACTOR times it, or None while neither test
    has settled and the replays are not ``final``.

    Counting replays against reads is conservative for the clean side: a replay
    is undecided only if one of its reads is."""
    above_bad = binomial_side_of_boundary(
        undecided, reads, EDGE_FACTOR * clean, failure_prob=failure_prob
    )
    if above_bad:
        return PROMOTE
    above_clean = binomial_side_of_boundary(
        undecided, reads, clean, failure_prob=failure_prob
    )
    if above_clean is False:
        return DROP
    if above_bad is False and above_clean:
        return ROLL_OVER
    if not final:
        return None
    return ROLL_OVER if undecided > clean * reads else DROP


def judge_edge(
    source, letter, sifter, *, clean, failure_prob
) -> Tuple[str, List[bytes]]:
    """Replay draws of ``source`` extended by ``letter`` through ``sifter`` until
    `edge_verdict` settles: the verdict, and the boundary strings the replays met."""
    met, undecided, reads = [], 0, 0
    for probe in range(1, EDGE_MAX_PROBES + 1):
        try:
            drawn = source.draw()
        except SourceDry:
            break
        before = sifter.reads
        leaf, boundary = sifter.sift_and_boundary(drawn + letter)
        reads += sifter.reads - before
        if leaf is None:
            undecided += 1
            met.append(boundary)
        verdict = edge_verdict(
            undecided,
            reads,
            clean,
            failure_prob=failure_prob,
            final=probe == EDGE_MAX_PROBES,
        )
        if verdict is not None:
            return verdict, met
    return (
        edge_verdict(undecided, reads, clean, failure_prob=failure_prob, final=True),
        met,
    )


class EdgeChain(RejectionSource):
    """The draws of ``parent`` whose extension by ``letter`` the round of
    ``sifter`` could not place."""

    def __init__(self, parent, letter, sifter, *, good, poor):
        super().__init__()
        self.parent = parent
        self.letter = letter
        self._sifter = sifter
        self._good = good
        self._poor = poor

    @property
    def proving(self) -> tuple:
        return proving_attempts(self._good, self._poor)

    @property
    def poor(self) -> float:
        return self._poor

    def attempt_draw(self) -> bool:
        drawn = self.parent.draw()
        if self._sifter.sift_and_boundary(drawn + self.letter)[0] is None:
            self._pool.append(drawn)
            return True
        return False

    def source_repr(self) -> str:
        return f"edge chain on {list(self.letter)}"
