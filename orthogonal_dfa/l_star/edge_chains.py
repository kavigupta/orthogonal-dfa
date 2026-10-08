"""Edges a round reads badly, found where the boundary population would dilute them.

An edge is a source of prefixes extended by a letter.  A target state the family
reads badly but that is rare among reads hides in the boundary population; an
edge leading into it finds it at its share among the source's draws, so the edge
is judged on its own: promoted to a population of its own, rolled over into a
chain the next round tests afresh, or dropped.  A chain keeps only the draws
whose extension its round's tree could not place, so each round it is rolled over
it is richer in the states leading into the badly read one -- provided each link
reads with suffixes no earlier link of the chain read, since a string's noise is
keyed by the string and a clean survivor would otherwise survive every link.
"""

from math import ceil, log
from typing import List, Optional, Tuple

from .cluster import read_rates, readable_size_and_margin, smallest_readable_family
from .rejection_source import RejectionSource, SourceDry, proving_attempts
from .sifting import Sifter
from .statistics import binomial_side_of_boundary
from .suffix_family import SuffixFamily

PROMOTE, ROLL_OVER, DROP = "promote", "roll over", "drop"

#: An edge undecided more than this many times the clean per-read bound is mostly
#: a badly read state: f > uHi / (uHi - 2 tau), which is 2 at the gap uHi = 4 tau.
EDGE_FACTOR = 3
#: Between a flat chain (its parent's rate) and one carrying a badly read state,
#: whose rate a fresh link at least doubles while that state is dilute.
EDGE_RISE = 1.5
#: Chance a round misjudges any of the edges it tests, split evenly over them.
EDGE_MISJUDGE = 0.01


def edge_verdict(
    undecided, reads, *, promote_above, keep_above, failure_prob, final
) -> Optional[str]:
    """Where an edge undecided on ``undecided`` replays over ``reads`` node reads
    stands against the two per-read rates, or None while neither test has settled
    and the replays are not ``final``.

    Counting replays against reads is conservative for the low side: a replay is
    undecided only if one of its reads is."""
    above_promote = binomial_side_of_boundary(
        undecided, reads, promote_above, failure_prob=failure_prob
    )
    if above_promote:
        return PROMOTE
    above_keep = binomial_side_of_boundary(
        undecided, reads, keep_above, failure_prob=failure_prob
    )
    if above_keep is False:
        return DROP
    if above_promote is False and above_keep:
        return ROLL_OVER
    if not final:
        return None
    return ROLL_OVER if undecided > keep_above * reads else DROP


def separating_reads(low, high, failure_prob) -> int:
    """Reads after which a rate at ``low`` or below and one at ``high`` or above
    are each misjudged with chance at most ``failure_prob`` (Chernoff, both read
    at the higher rate's variance); where the two are close, as close as a chain
    and its rise."""
    low, high = sorted((low, high))
    gap = max(high - low, high * (1 - 1 / EDGE_RISE))
    return ceil(2 * high * log(1 / failure_prob) / gap**2)


def judge_edge(
    source, letter, sifter, *, promote_above, keep_above, failure_prob
) -> Tuple[str, List[bytes], float]:
    """Replay draws of ``source`` extended by ``letter`` through ``sifter`` until
    `edge_verdict` settles, or until `separating_reads` have been made: the
    verdict, the boundary strings the replays met, and the undecided rate per
    read."""
    cap = separating_reads(keep_above, promote_above, failure_prob)
    met, undecided, reads = [], 0, 0
    while reads < cap:
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
            promote_above=promote_above,
            keep_above=keep_above,
            failure_prob=failure_prob,
            final=reads >= cap,
        )
        if verdict is not None:
            return verdict, met, undecided / reads
    verdict = edge_verdict(
        undecided,
        reads,
        promote_above=promote_above,
        keep_above=keep_above,
        failure_prob=failure_prob,
        final=True,
    )
    return verdict, met, undecided / max(reads, 1)


def fresh_sifter(pst, tree, vs, used) -> Optional[Tuple[Sifter, frozenset]]:
    """A sifter over ``tree`` reading the round family ``vs``'s rows not in
    ``used``, topped up from the rows its coherent group ranks next, at the band their
    number reads, and the rows it reads; None where the group runs out first.

    Rows of one group vote alike, so a link reads the round's cut on rows no
    earlier link read.  The empty suffix is left out: a link's reads must be
    fresh strings, and every chain's draws were read with it."""
    boundary = pst.decision_boundary
    signal = pst.config.min_signal_strength
    rates = read_rates(pst, boundary)
    smallest = smallest_readable_family(signal, boundary, rates)
    rows = [
        v
        for v in [*vs, *(v for v in pst.suffix_group if v not in vs)]
        if v not in used and pst.table.suffix(v) != b""
    ]
    if len(rows) < smallest:
        return None
    size, margin = readable_size_and_margin(
        signal, boundary, len(rows), smallest, rates
    )
    family = SuffixFamily(pst, rows[:size])
    family.accept_thresh = boundary + margin
    family.reject_thresh = boundary - margin
    return Sifter(tree, family), frozenset(rows[:size])


class EdgeChain(RejectionSource):
    """The draws of ``parent`` whose extension by ``letter`` ``sifter`` could not
    place; ``used`` is every suffix row this link and those before it read, and
    ``rate`` the undecided rate per read that rolled it over."""

    def __init__(self, parent, letter, sifter, *, used, rate):
        super().__init__()
        self.parent = parent
        self.letter = letter
        self.used = used
        self.rate = rate
        self._sifter = sifter

    @property
    def proving(self) -> tuple:
        return proving_attempts(min(1.0, 2 * self.rate), self.rate)

    @property
    def poor(self) -> float:
        # Per read, so at most what an attempt yields.
        return self.rate

    def attempt_draw(self) -> bool:
        drawn = self.parent.draw()
        if self._sifter.sift_and_boundary(drawn + self.letter)[0] is None:
            self._pool.append(drawn)
            return True
        return False

    def source_repr(self) -> str:
        return f"edge chain on {list(self.letter)}"
