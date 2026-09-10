"""Drawing the prefixes that belong to one state.

The hypothesis says where to aim a string; the tree says where it landed, and
only the tree's answer counts.
"""

import math
from typing import Optional

from .dfa_utils import count_paths_to_state, sample_string_reaching_state

#: A source landing fewer of its aims than this is one to stop waiting for.  At
#: yield ``p`` it takes about ``1 / p`` draws per prefix, which is what bounds
#: the asking.  A floor on patience, not a measurement.
MIN_YIELD = 0.2


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
        self._population.add(aimed)
        # Where it rests, not where it was aimed.
        if self._population.settle(aimed, self._path):
            self._pool.append(aimed)
            return True
        return False

    def aims_land(self) -> bool:
        """Whether aiming at this leaf lands often enough to keep asking.

        At yield ``MIN_YIELD`` one lands within ``1 / MIN_YIELD`` aims more often
        than not, so that many is what the leaf gets to prove itself in.  A floor
        on patience, like the yield itself, not a measurement of it.
        """
        return any(self.aimed_draw() for _ in range(math.ceil(1 / MIN_YIELD)))

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
            for _ in range(math.ceil(1 / MIN_YIELD)):
                self.aimed_draw()
            # Landing is not enough: a leaf whose support is spent goes on
            # landing strings it has already given, and waiting for a new one
            # would be waiting forever.
            if all(member in self._served for member in self._pool):
                return None


def gather(source, wanted: int) -> list:
    """Up to ``wanted`` distinct prefixes from ``source``, however few it gives.

    Fewer is the source saying so: at ``MIN_YIELD`` this many asks is what
    ``wanted`` costs, and a source that cannot fill it in that is one the round
    cannot read a rate over.
    """
    held = set()
    for _ in range(math.ceil(wanted / MIN_YIELD)):
        if len(held) == wanted:
            break
        drawn = source.draw()
        if drawn is not None:
            held.add(drawn)
    return sorted(held)
