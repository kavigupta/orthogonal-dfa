"""Drawing the prefixes that belong to one state.

A source belongs to the round that made it: it closes over that round's tree,
which says where a string rests, and its hypothesis, which says where to aim
one.  Only the tree's answer counts.
"""

from math import ceil, isqrt, log

import scipy.stats

from .dfa_utils import (
    count_paths_to_state,
    sample_string_reaching_state,
    uniform_weights,
)
from .sifting import anchored_walk, first_disagreeing_edge
from .statistics import binom_cdf

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


#: A probe stream stranding at least this share is one worth drawing from.  Far
#: under a leaf's bars: aiming either lands or does not, where a probe is only
#: asked to turn up something the family cannot place, which is rarer and enough.
GOOD_STRAND = 0.2
POOR_STRAND = 0.1


class _Source:
    """A pool filled by aiming, served until aiming stops filling it.

    A subclass says what an aim is: `aimed_draw` makes one, puts what it found
    in ``_pool``, and says whether any of that was new.  ``PROVING`` is the
    ``(aims, kept)`` that test is sized to and ``POOR`` the yield below which a
    source is not worth asking again -- both from `_proving_aims`.
    """

    PROVING: tuple
    POOR: float

    def __init__(self, served=()):
        #: Handed out already.  An aim that rests on one of these has landed but
        #: has nothing to serve, so yield alone never says a source is spent.
        self._served = set(served)
        self._pool = []

    def aimed_draw(self) -> bool:
        raise NotImplementedError

    def source_repr(self) -> str:
        """Which source this is, for the error when it runs dry."""
        raise NotImplementedError

    def has_sufficient_yield(self) -> bool:
        """Whether aims land often enough to keep making them."""
        aims, kept = self.PROVING
        return sum(self.aimed_draw() for _ in range(aims)) > kept

    def draw(self, false_alarm_p=1e-9) -> bytes:
        """One string from the pool, aiming for more when it runs dry."""
        #: Aims in a row that turn up nothing new before a source is called dry.
        #: Geometric distribution.
        dry = ceil(log(false_alarm_p) / log(1 - self.POOR))
        # One pass more than the aims it counts: what an aim turned up is read
        # by the drain of the pass after it.
        for _ in range(dry + 1):
            while self._pool:
                member = self._pool.pop()
                if member not in self._served:
                    self._served.add(member)
                    return member
            self.aimed_draw()
        raise RuntimeError(
            f"Source {self.source_repr()} found no new samples in {dry} attempts"
        )


class BoundarySource(_Source):
    """Strings the round's tree could not place, found the way the round finds
    them: a probe, anchored where the tree first places a prefix, walked through
    the round's transitions, and bisected where the walk and a fresh sift
    disagree.

    Every sift on the way is a question the family may not answer, and the ones
    it does not are what this yields.  Its only supply is the sampler, so unlike
    a source aimed at one state it can never be short of input -- a thin state
    has finitely many members, a probe stream has none.
    """

    PROVING = _proving_aims(GOOD_STRAND, POOR_STRAND)
    POOR = POOR_STRAND

    def __init__(self, pst, sifter, transitions, *, known=(), label=("boundary",)):
        super().__init__(served=known)
        self.label = label
        self._pst = pst
        self._sifter = sifter
        self._transitions = transitions
        #: Every string this source has produced or been told about.  A probe
        #: strands the same one as often as not -- the walk starts at the empty
        #: prefix, so a tree that cannot place that cannot place it for any probe
        #: -- and a repeat is not something to have found.
        self._seen = set(known)

    def _sift(self, seq):
        """Sift, keeping what the family could not answer for."""
        leaf, boundary = self._sifter.sift_and_boundary(seq)
        if leaf is None and boundary not in self._seen:
            self._seen.add(boundary)
            self._pool.append(boundary)
        return leaf

    def aimed_draw(self) -> bool:
        """One probe, walked the way a round walks it.  Says whether anything
        the family could not answer came of it *that it had not already found*:
        stranding the same string again is not a draw this source can serve.
        """
        before = len(self._seen)
        probe = self._pst.sampler.sample(
            self._pst.rng, alphabet_size=self._pst.alphabet_size
        )
        start, states = anchored_walk(probe, self._sift, self._transitions)
        if start is not None:
            landed = self._sift(probe)
            if landed is not None and landed != states[-1]:
                # The edge narrowed to is thrown away -- the round has already
                # had it.  What is wanted is the prefixes asked about on the
                # way, which `_sift` keeps.
                first_disagreeing_edge(probe, states, self._sift, start, len(probe))
        return len(self._seen) > before

    def source_repr(self) -> str:
        return str(self.label)


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


class StateSource(_Source):
    """Prefixes the tree places at one leaf."""

    PROVING = _proving_aims(GOOD_YIELD, POOR_YIELD)
    POOR = POOR_YIELD

    def __init__(self, resolver, leaf, aim, *, wanted):
        super().__init__()
        self._population = resolver.population
        self._path = resolver.tree.path_of(leaf)
        # A split replaces a leaf with a node holding both ids, so every id the
        # tree reports has a path to it.
        assert self._path is not None, leaf
        self._aim = aim
        #: Resting at the leaf and not yet handed out.  Reading a leaf pushes
        #: strings down to it, so the count is work rather than a cap: ask for
        #: what this source is being built to serve.
        self._pool.extend(self._population.members(self._path, wanted))

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

    def source_repr(self) -> str:
        return f"leaf {self._path}"
