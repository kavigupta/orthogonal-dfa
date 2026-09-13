"""Where each population of prefixes comes from.

A source belongs to the round that made it: it closes over that round's tree,
which says where a string rests, and its hypothesis, which says where to aim
one.  Only the tree's answer counts.
"""

from math import ceil, isqrt, log
from typing import Optional

import scipy.stats

from .dfa_utils import (
    count_paths_to_state,
    sample_string_reaching_state,
    uniform_weights,
)
from .mask_table import UNIFORM
from .statistics import binom_cdf

#: Prefixes a population is asked for.
WANTED = 100
#: A leaf landing at least this share of its aims is one worth asking again.
GOOD_YIELD = 0.5
#: One landing at most this share is one to stop asking.  Nothing decides
#: anything between the two bars, which is what keeps the count that tells them
#: apart affordable.
POOR_YIELD = 0.25
#: Chance of reading a leaf as either bar when it is the other.
_MISREAD = 1e-5


def _proving_aims(good, poor):
    """Sizes the test so that P(reject | yield >= ``good``) and
    P(keep | yield <= ``poor``) are both bounded by ``_MISREAD``."""
    aims = 0
    while True:
        aims += 1
        # The fewest landings a poor leaf is unlikely to reach; a good one has
        # to clear it for the same count to answer both questions.
        landings = int(scipy.stats.binom.isf(_MISREAD, aims, poor))
        if binom_cdf(landings, aims, good) <= _MISREAD:
            return aims, landings


PROVING_AIMS, LANDINGS_KEPT = _proving_aims(GOOD_YIELD, POOR_YIELD)
#: A probe stream stranding at least this share is one worth drawing from.  Far
#: under a leaf's bars: aiming either lands or does not, where a probe is only
#: asked to turn up something the family cannot place, which is rarer and enough.
GOOD_STRAND = 0.2
POOR_STRAND = 0.1
PROBES, STRANDS_KEPT = _proving_aims(GOOD_STRAND, POOR_STRAND)


class UniformSource:
    """The learner's own sampler.  Every draw is a prefix, so this never fails.

    Labelled ``UNIFORM``: it draws for the pool the table already keeps, and two
    populations of one sampler would be one vote each however differently they
    had grown.
    """

    label = UNIFORM

    def __init__(self, pst):
        self._pst = pst

    def draw(self) -> Optional[bytes]:
        return self._pst.sampler.sample(
            self._pst.rng, alphabet_size=self._pst.alphabet_size
        )


class BoundarySource:
    """Strings the round's tree could not place, found the way the round finds
    them: a probe, anchored where the tree first places a prefix, walked through
    the round's transitions, and bisected where the walk and a fresh sift
    disagree.

    Every sift on the way is a question the family may not answer, and the ones
    it does not are what this yields.  Its only supply is the sampler, so unlike
    a source aimed at one state it can never be short of input -- a thin state
    has finitely many members, a probe stream has none.
    """

    def __init__(self, pst, sifter, transitions, label=("boundary",)):
        self.label = label
        self._pst = pst
        self._sifter = sifter
        self._transitions = transitions
        self._served = set()
        #: Every string this source has produced.  A probe strands the same one
        #: as often as not -- the walk starts at the empty prefix, so a tree that
        #: cannot place that cannot place it for any probe -- and a repeat is not
        #: something to have found.
        self._seen = set()
        self._pool = []

    def _sift(self, seq):
        """Sift, keeping what the family could not answer for."""
        leaf, boundary = self._sifter.sift_and_boundary(seq)
        if leaf is None and boundary not in self._seen:
            self._seen.add(boundary)
            self._pool.append(boundary)
        return leaf

    def aimed_draw(self) -> bool:
        """One probe, sifted the way a round sifts it.  Says whether anything
        the family could not answer came of it *that it had not already found*:
        stranding the same string again is not a draw this source can serve.
        """
        before = len(self._seen)
        probe = self._pst.sampler.sample(
            self._pst.rng, alphabet_size=self._pst.alphabet_size
        )
        start, state = 0, None
        while start < len(probe):
            state = self._sift(probe[:start])
            if state is not None:
                break
            start += 1
        if state is not None:
            states = [None] * start + [state]
            for symbol in probe[start:]:
                state = self._transitions[state][symbol]
                states.append(state)
            landed = self._sift(probe)
            if landed is not None and states[-1] is not None and landed != states[-1]:
                self._bisect(probe, states, start)
        return len(self._seen) > before

    def _bisect(self, probe, states, lo):
        """Narrow to the edge the walk and the sift disagree over.

        The answer is not wanted -- the round has already had it -- but the
        prefixes asked about on the way are the ones a family would have to
        answer for a round to act on this probe at all, and those are worth
        keeping.
        """
        hi = len(probe)
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            landed = self._sift(probe[:mid])
            if landed is None:
                return
            lo, hi = (mid, hi) if landed == states[mid] else (lo, mid)

    def has_sufficient_yield(self) -> bool:
        """Whether probes strand often enough to keep drawing them."""
        stranded = sum(self.aimed_draw() for _ in range(PROBES))
        return stranded > STRANDS_KEPT

    def draw(self, false_alarm_p=1e-9) -> bytes:
        """One string the tree could not place, probing until there is one."""
        dry = ceil(log(false_alarm_p) / log(1 - POOR_STRAND))
        for _ in range(dry + 1):
            while self._pool:
                member = self._pool.pop()
                if member not in self._served:
                    self._served.add(member)
                    return member
            self.aimed_draw()
        raise RuntimeError(
            f"{self.label} stranded nothing new in {dry} probes "
            f"after serving {len(self._served)}"
        )


def aim_at(pst, dfa, leaf):
    """
    Attempt to aim at a leaf, returning a source that draws on it or ``None`` if the leaf is
    unreachable from the starting state, or if too few strings of the sampler's
    length reach it to draw from without redrawing what it has already drawn.

    Aims are drawn with replacement, so a leaf that holds

        sqrt(alphabet_size ** length)

    of the strings of that length repeats one only after the fourth root of
    them have been drawn.  Scaled to the space rather than to the draws asked
    for: a leaf is thin compared to what the sampler could have put there.
    """
    weights = pst.sampler.symbol_weights(pst.alphabet_size)
    length = pst.sampler.length
    # Counted evenly rather than off the mass below: the sampler's weights are
    # read as ratios, so its mass is on no scale a threshold could name.
    reaching = count_paths_to_state(dfa, leaf, length, uniform_weights(dfa))
    if reaching[length][dfa.initial_state] < isqrt(pst.alphabet_size**length):
        return None
    mass = count_paths_to_state(dfa, leaf, length, weights)
    # A symbol the sampler never places can leave a well-reached leaf with none.
    if mass[length][dfa.initial_state] == 0:
        return None
    return lambda: sample_string_reaching_state(dfa, mass, pst.rng, weights)


def state_source(resolver, leaf, aim, *, wanted):
    """
    A source that draws on `aim` and guarantees (with probability 1 - _MISREAD)
    that at least POOR_YIELD (25%) of the strings it draws will land
    at the given leaf, according to the tree in `resolver`.

    If this guarantee cannot be made, returns None
    """
    source = StateSource(resolver, leaf, aim, wanted=wanted)
    return source if source.has_sufficient_yield() else None


class StateSource:
    """Prefixes the tree places at one leaf."""

    def __init__(self, resolver, leaf, aim, *, wanted):
        self.label = ("state", leaf)
        self._population = resolver.population
        self._path = resolver.tree.path_of(leaf)
        # A split replaces a leaf with a node holding both ids, so every id the
        # tree reports has a path to it.
        assert self._path is not None, leaf
        self._aim = aim
        self._served = set()
        #: Resting at the leaf and not yet handed out.  Reading a leaf pushes
        #: strings down to it, so the count is work rather than a cap: ask for
        #: what this source is being built to serve.
        self._pool = list(self._population.members(self._path, wanted))

    def aimed_draw(self) -> bool:
        """Aim one string, let the tree place it, and say whether it rested here.

        One that does joins the pool.  One that does not belongs to the leaf it
        did rest at, which is the answer that counts.
        """
        aimed = self._aim()
        # Where it rests, not where it was aimed.
        if self._population.settle(aimed, self._path):
            self._pool.append(aimed)
            return True
        return False

    def has_sufficient_yield(self) -> bool:
        """
        Check whether there is sufficient yield.
        """
        landed = sum(self.aimed_draw() for _ in range(PROVING_AIMS))
        return landed > LANDINGS_KEPT

    def draw(self, false_alarm_p=1e-9) -> bytes:
        """Provide a prefix resting at the leaf, aiming for more when the pool runs
        dry."""
        #: Aims in a row that rest nothing new before the leaf is called dry. Geometric distribution.
        dry_aims = ceil(log(false_alarm_p) / log(1 - POOR_YIELD))

        # One pass more than the aims it counts: what an aim landed is read by
        # the drain of the pass after it.
        for _ in range(dry_aims + 1):
            while self._pool:
                member = self._pool.pop()
                if member not in self._served:
                    self._served.add(member)
                    return member
            self.aimed_draw()
        # An aim that rests where it was aimed on a string already served counts
        # as landing, so yield alone never says a leaf is spent.
        raise RuntimeError(
            f"leaf {self._path} rested nothing new in {dry_aims} aims "
            f"after serving {len(self._served)}"
        )

    def unused(self, drawn) -> None:
        """Take back draws a collection did not use.  Landing one is the
        expensive part and the leaf is still where it rests, so it goes back in
        the pool rather than being served twice over."""
        self._served.difference_update(drawn)
        self._pool.extend(drawn)


def gather(source, wanted: int) -> list:
    """Up to ``wanted`` distinct prefixes from ``source``, however few it gives.

    For growing a population, where any is a gain.  Defining one is ``collect``,
    which holds out for the whole number.
    """
    held = set()
    for _ in range(ceil(wanted / POOR_YIELD)):
        if len(held) == wanted:
            break
        drawn = source.draw()
        if drawn is not None:
            held.add(drawn)
    return sorted(held)


def collect(source, wanted: int) -> Optional[list]:
    """``wanted`` distinct prefixes from ``source``, or ``None`` if it could not.

    Giving up is the point: a population nothing can be drawn for is one the
    round cannot read a rate over, and saying so beats holding it to one.
    """
    held = gather(source, wanted)
    if len(held) == wanted:
        return held
    # Taking is only earned by a population coming of it.
    give_back = getattr(source, "unused", None)
    if give_back is not None:
        give_back(held)
    return None


def draw_for_split(source, wanted: int) -> list:
    """Prefixes from ``source`` to read the split on and nothing else.

    A population that cannot certify on the prefixes it holds needs more of its
    own -- and read only for the split, since adding them to the table costs a
    query on every fully observed column and unsettles the FNR the round has
    just met.
    """
    return gather(source, wanted)
