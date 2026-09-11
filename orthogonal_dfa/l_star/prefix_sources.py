"""Drawing the prefixes that belong to one state.

The hypothesis says where to aim a string; the tree says where it landed, and
only the tree's answer counts.
"""

import math
from typing import Optional

import scipy.stats

from .dfa_utils import count_paths_to_state, sample_string_reaching_state
from .statistics import _binom_cdf

#: A leaf landing at least this share of its aims is one worth asking again.
GOOD_YIELD = 0.5
#: One landing at most this share is one to stop asking.  Between the two bars
#: either answer will do, and that indifference is what keeps the reading short:
#: telling apart rates that close would take an unaffordable number of aims.
POOR_YIELD = 0.25
#: Chance of reading a leaf as either bar when it is the other.
_MISREAD = 1e-5


def _proving_aims():
    """``(aims, landings)``: how many aims read a leaf's yield, and how many of
    them it has to land to be kept.

    The fewest aims at which a leaf at ``GOOD_YIELD`` is kept and one at
    ``POOR_YIELD`` dropped, each but for ``_MISREAD``.
    """
    aims = 0
    while True:
        aims += 1
        # The fewest landings a poor leaf is unlikely to reach; a good one has
        # to clear it for the same count to answer both questions.
        landings = int(scipy.stats.binom.isf(_MISREAD, aims, POOR_YIELD))
        if _binom_cdf(landings, aims, GOOD_YIELD) <= _MISREAD:
            return aims, landings


PROVING_AIMS, LANDINGS_KEPT = _proving_aims()


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


def state_source(resolver, leaf, aim, *, wanted):
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
    source = StateSource(resolver, leaf, aim, wanted=wanted)
    return source if source.aims_land() else None


class StateSource:
    """Prefixes the tree places at one leaf."""

    def __init__(self, resolver, leaf, aim, *, wanted):
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


def gather(source, wanted: int) -> list:
    """Up to ``wanted`` distinct prefixes from ``source``, however few it gives.

    Fewer is the source saying so: at ``POOR_YIELD`` this many asks is what
    ``wanted`` costs, and a source that cannot fill it in that is one the round
    cannot read a rate over.
    """
    held = set()
    for _ in range(math.ceil(wanted / POOR_YIELD)):
        if len(held) == wanted:
            break
        drawn = source.draw()
        if drawn is not None:
            held.add(drawn)
    return sorted(held)
