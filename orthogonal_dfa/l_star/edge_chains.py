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
#: Replays an edge gets before it is judged on its point estimate.
EDGE_MAX_PROBES = 300


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


def judge_edge(
    source, letter, sifter, *, promote_above, keep_above, failure_prob
) -> Tuple[str, List[bytes], float]:
    """Replay draws of ``source`` extended by ``letter`` through ``sifter`` until
    `edge_verdict` settles: the verdict, the boundary strings the replays met, and
    the undecided rate per read."""
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
            promote_above=promote_above,
            keep_above=keep_above,
            failure_prob=failure_prob,
            final=probe == EDGE_MAX_PROBES,
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
    """A sifter over ``tree`` reading only the suffix rows of ``vs`` not in
    ``used``, at the band their number reads, and the rows it reads; None where
    too few are left to read a decision."""
    boundary = pst.decision_boundary
    signal = pst.config.min_signal_strength
    rates = read_rates(pst, boundary)
    fresh = [v for v in vs if v not in used]
    smallest = smallest_readable_family(signal, boundary, rates)
    if len(fresh) < smallest:
        return None
    size, margin = readable_size_and_margin(
        signal, boundary, len(fresh), smallest, rates
    )
    family = SuffixFamily(pst, fresh[:size])
    family.accept_thresh = boundary + margin
    family.reject_thresh = boundary - margin
    return Sifter(tree, family), frozenset(fresh[:size])


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
