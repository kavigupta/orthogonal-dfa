"""Where each population of prefixes comes from.

A source belongs to the round that made it: it closes over that round's tree,
which says where a string rests, and its hypothesis, which says where to aim
one.  A later round asking the same source gets the earlier round's answers.
"""

import math
from collections import deque
from typing import Optional

from .dfa_utils import count_paths_to_state, sample_string_reaching_state
from .mask_table import UNIFORM
from .statistics import _binom_cdf

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


#: Where ``Binomial(n, POOR_YIELD)`` and ``Binomial(n, GOOD_YIELD)`` cross, as a
#: share of the aims.  The count they are read over has to be searched for, but
#: the line between them does not: it is the same share whatever the count.
_KEPT_ABOVE = math.log((1 - POOR_YIELD) / (1 - GOOD_YIELD)) / math.log(
    GOOD_YIELD * (1 - POOR_YIELD) / (POOR_YIELD * (1 - GOOD_YIELD))
)


def _proving_aims():
    """The fewest aims at which a leaf at ``GOOD_YIELD`` is kept and one at
    ``POOR_YIELD`` dropped, each but for ``_MISREAD``, and the landings that
    separate them.

    Searched rather than solved.  Exact binomial tails have no closed form for
    the count, and the count that works is not monotone -- 264 aims answer both
    questions and 265 do not -- so there is nothing here to bisect.
    """
    aims = 0
    while True:
        aims += 1
        landings = math.floor(_KEPT_ABOVE * aims)
        if (
            1 - _binom_cdf(landings, aims, POOR_YIELD) <= _MISREAD
            and _binom_cdf(landings, aims, GOOD_YIELD) <= _MISREAD
        ):
            return aims, landings


PROVING_AIMS, LANDINGS_KEPT = _proving_aims()


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
    """Strings the round's tree could not place, drawn the way that round found
    them: a probe, then the prefixes of it, sifted until one lands.

    The sifter is the round's, not the caller's.  A pool is the strings *that*
    tree straddled, so growing it means asking that tree again -- against a
    later one the same draw would be a different population.
    """

    def __init__(self, pst, sifter, label):
        self.label = label
        self._pst = pst
        self._sifter = sifter
        self._served = set()
        self._pending = deque()

    def draw(self) -> Optional[bytes]:
        """One unplaced string from a fresh probe, or ``None`` if it had none.

        A probe usually strands more than one, so the rest are kept for the next
        ask rather than resifted.
        """
        if not self._pending:
            self._sift_a_probe()
        while self._pending:
            drawn = self._pending.popleft()
            if drawn not in self._served:
                self._served.add(drawn)
                return drawn
        return None

    def _sift_a_probe(self) -> None:
        probe = self._pst.sampler.sample(
            self._pst.rng, alphabet_size=self._pst.alphabet_size
        )
        # The prefix walk stops where the tree first places one, and the whole
        # probe is read after it: the two points a round sifts at.
        for start in range(len(probe) + 1):
            leaf, boundary = self._sifter.sift_and_boundary(probe[:start])
            if leaf is not None:
                break
            self._pending.append(boundary)
        leaf, boundary = self._sifter.sift_and_boundary(probe)
        if leaf is None:
            self._pending.append(boundary)


def aim_at(pst, dfa, leaf):
    """A callable drawing strings the hypothesis says reach ``leaf``, or ``None``
    where it has none of the sampler's length that do -- no path, or none its
    symbol weights would take.

    The callable always draws: what it refuses is the mass being zero, and that
    is what ``None`` here reports instead.
    """
    weights = pst.sampler.symbol_weights(pst.alphabet_size)
    length = pst.sampler.length
    mass = count_paths_to_state(dfa, leaf, length, weights)
    if not mass[length][dfa.initial_state]:
        return None
    return lambda: sample_string_reaching_state(dfa, mass, pst.rng, weights)


def state_source(resolver, leaf, aim, *, wanted, sink):
    """A source drawing on ``aim``, or ``None`` where the tree does not rest what
    it draws at ``leaf``.

    The hypothesis says where to aim and the tree says where it lands, and the
    two disagree.  A leaf almost nothing settles at has only what already rests
    there to give, which runs out -- a finite population wearing an infinite
    one's clothes -- so it is probed before it is kept.

    Whether the hypothesis can aim there at all is ``aim_at``'s answer, asked
    first: that one is about the leaf being out of reach rather than about
    anything the round did.
    """
    source = StateSource(resolver, leaf, aim, wanted=wanted, sink=sink)
    return source if source.aims_land() else None


class StateSource:
    """Prefixes the tree places at one leaf."""

    def __init__(self, resolver, leaf, aim, *, wanted, sink):
        self.label = ("state", leaf)
        self._population = resolver.population
        self._path = resolver.tree.path_of(leaf)
        # A split replaces a leaf with a node holding both ids, so every id the
        # tree reports has a path to it.
        assert self._path is not None, leaf
        self._aim = aim
        self._sink = sink
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
        if self._population.resting_at(aimed) is None:
            self._sink(aimed)
        return False

    def aims_land(self) -> bool:
        """Whether this leaf lands enough of ``PROVING_AIMS`` aims to keep asking.

        Read over a count sized to answer it, rather than guessed at from a
        handful.  Nothing is wasted on a leaf that passes: an aim is a string
        pushed at the leaf either way, and the ones that land are in the pool
        before the first draw asks for one.
        """
        landed = sum(self.aimed_draw() for _ in range(PROVING_AIMS))
        return landed > LANDINGS_KEPT

    def draw(self) -> Optional[bytes]:
        """One prefix resting at the leaf, or ``None`` where a round of aiming
        brought back nothing the leaf has not already given.

        Which covers both a run of misses and a leaf whose whole reachable
        support is spent.  The two are not worth telling apart here: either way
        this ask got nothing, and how hard to keep trying is the caller's budget
        to spend, not this one's.
        """
        while True:
            while self._pool:
                member = self._pool.pop()
                if member not in self._served:
                    self._served.add(member)
                    return member
            for _ in range(math.ceil(1 / POOR_YIELD)):
                self.aimed_draw()
            # Landing is not enough: a leaf whose support is spent goes on
            # landing strings it has already given, and waiting for a new one
            # would be waiting forever.
            if all(member in self._served for member in self._pool):
                return None

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
    for _ in range(math.ceil(wanted / POOR_YIELD)):
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
