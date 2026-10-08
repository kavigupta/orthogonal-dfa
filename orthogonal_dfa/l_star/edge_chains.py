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
from typing import Optional, Tuple

from .cluster import read_rates, readable_size_and_margin, smallest_readable_family
from .rejection_source import RejectionSource, SourceDry, proving_attempts
from .sifting import Sifter
from .statistics import binomial_side_of_boundary
from .suffix_family import SuffixFamily

PROMOTE, ROLL_OVER, DROP = "promote", "roll over", "drop"
UNDECIDED_EDGE, DISAGREEMENT_EDGE = "undecided", "disagreement"

#: An edge undecided more than this many times the clean per-read bound is mostly
#: a badly read state: f > uHi / (uHi - 2 tau), which is 2 at the gap uHi = 4 tau.
EDGE_FACTOR = 3
#: Between a flat chain (its parent's rate) and one carrying a badly read state,
#: whose rate a fresh link at least doubles while that state is dilute.
EDGE_RISE = 1.5
#: Chance a round misjudges any of the edges it tests, split evenly over them.
EDGE_MISJUDGE = 0.01
#: A decided read's chance of landing on the wrong side (about 1e-14), raised to a
#: rate a capped test can tell from a disagreement it is worth rolling over.
CLEAN_FLIP = 1e-6


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


def detecting_reads(rate, failure_prob) -> int:
    """Reads after which an edge at ``rate`` or above has shown a hit but for
    chance ``failure_prob``: (1 - rate)^n <= exp(-n rate)."""
    return ceil(log(1 / failure_prob) / rate)


def separating_reads(low, high, failure_prob) -> int:
    """Reads after which a rate at ``low`` or below and one at ``high`` or above
    are each misjudged with chance at most ``failure_prob`` (Chernoff, both read
    at the higher rate's variance); where the two are close, as close as a chain
    and its rise."""
    low, high = sorted((low, high))
    gap = max(high - low, high * (1 - 1 / EDGE_RISE))
    return ceil(2 * high * log(1 / failure_prob) / gap**2)


def undecided_measure(sifter, letter):
    """For an undecided edge: whether ``sifter`` cannot place a draw extended by
    ``letter``, the node reads that took, and the string it could not place."""

    def measure(drawn):
        before = sifter.reads
        leaf, boundary = sifter.sift_and_boundary(drawn + letter)
        return leaf is None, sifter.reads - before, [boundary] if leaf is None else []

    return measure


def midpoint_disagreement(family, tree, transitions, boundary, letter):
    """For a disagreement edge: whether a draw extended by ``letter``, read at the
    middle of the band as `estimate_agreement_rate` reads it, lands somewhere other
    than the hypothesis's edge out of the draw's own leaf; one replay each, and
    the draw itself, which is the member of the leaf the split test needs."""

    def decide(seq, midfix):
        return family.mean(seq, midfix) >= boundary

    def measure(drawn):
        leaf = tree.classify(drawn, decide)
        landed = tree.classify(drawn + letter, decide)
        hit = (
            leaf is not None
            and landed is not None
            and landed != transitions[leaf][letter[0]]
        )
        return hit, 1, [drawn] if hit else []

    return measure


class EdgeTest:
    """One edge's running tally against its two rates, settled by `edge_verdict`
    or at ``cap`` units of what ``measure`` weighs a replay at; dropped at
    ``screen`` units with no hit, where that is not None."""

    def __init__(
        self, measure, *, promote_above, keep_above, failure_prob, cap, screen
    ):
        self.measure = measure
        self.promote_above = promote_above
        self.keep_above = keep_above
        self.failure_prob = failure_prob
        self.cap = cap
        self.screen = screen
        self.hits = 0
        self.weight = 0
        self.found = []
        self.verdict = None

    @property
    def rate(self) -> float:
        return self.hits / max(self.weight, 1)

    def record(self, drawn) -> None:
        hit, cost, got = self.measure(drawn)
        self.hits += hit
        self.weight += cost
        self.found += got
        if self.screen is not None and self.hits == 0 and self.weight >= self.screen:
            self.verdict = DROP
            return
        self.settle(final=self.weight >= self.cap)

    def settle(self, *, final) -> None:
        self.verdict = edge_verdict(
            self.hits,
            self.weight,
            promote_above=self.promote_above,
            keep_above=self.keep_above,
            failure_prob=self.failure_prob,
            final=final,
        )


def judge_edges(source, tests) -> None:
    """Replay draws of ``source`` until every test of ``tests`` has a verdict,
    each draw read by every test still open: one draw serves every letter and
    kind of edge out of the source."""
    while True:
        open_tests = [test for test in tests if test.verdict is None]
        if not open_tests:
            return
        try:
            drawn = source.draw()
        except SourceDry:
            for test in open_tests:
                test.settle(final=True)
            return
        for test in open_tests:
            test.record(drawn)


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
    """The draws of ``parent`` that ``measure`` (read on the extension by
    ``letter``) hits; ``used`` is every suffix row this link and those before it
    read, and ``rate`` the hit rate that rolled it over."""

    def __init__(self, parent, letter, measure, *, kind, used, rate):
        super().__init__()
        self.parent = parent
        self.letter = letter
        self.kind = kind
        self.used = used
        self.rate = rate
        self._measure = measure

    @property
    def proving(self) -> tuple:
        return proving_attempts(min(1.0, 2 * self.rate), self.rate)

    @property
    def poor(self) -> float:
        # Per unit the measure weighs, so at most what an attempt yields.
        return self.rate

    def attempt_draw(self) -> bool:
        drawn = self.parent.draw()
        if self._measure(drawn)[0]:
            self._pool.append(drawn)
            return True
        return False

    def source_repr(self) -> str:
        return f"{self.kind} chain on {list(self.letter)}"
