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


def _aim_at(pst, dfa, leaf):
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


def state_source(pst, resolver, dfa, leaf, *, wanted):
    """A source for ``leaf``, or ``None`` where aiming at it does not land.

    Two ways it does not.  The hypothesis may have no string of the sampler's
    length reaching the leaf at all, and then there is nothing to aim.  Or it may
    have plenty and the tree place almost none of them there -- the hypothesis
    says where to aim, the tree says where it lands, and they disagree.

    Either way the leaf has only what already rests at it, and no draw of the
    sampler's adds to that, so the round does not come up short on it.
    """
    aim = _aim_at(pst, dfa, leaf)
    if aim is None:
        return None
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
        self._wanted = wanted
        self._served = set()
        #: Resting at the leaf and not yet handed out.  Aiming is how it refills,
        #: so what a probe lands is already in it before the first draw.
        self._pool = []
        self._read_the_leaf = False

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
        """One prefix resting at the leaf, or ``None`` once it has no more.

        What the pool does not hold it aims for, so running out means the leaf's
        whole reachable support has been served -- not that a draw missed.
        """
        while True:
            while self._pool:
                member = self._pool.pop()
                if member not in self._served:
                    self._served.add(member)
                    return member
            if not self._read_the_leaf:
                self._read_the_leaf = True
                # Reading a leaf pushes strings down to it, so the count is
                # work rather than a cap: ask for what could still be served.
                self._pool.extend(
                    self._population.members(
                        self._path, len(self._served) + self._wanted
                    )
                )
                continue
            # Aims that only bring back what has been served already are the
            # leaf saying it has nothing else, which is different from missing.
            if not any(self.aimed_draw() for _ in range(math.ceil(1 / MIN_YIELD))):
                return None


def collect(source, wanted: int) -> Optional[list]:
    """``wanted`` distinct prefixes from ``source``, or ``None`` if it could not.

    Giving up is the point: a population nothing can be drawn for is one the
    round cannot read a rate over, and saying so beats holding it to one.
    """
    held = set()
    for _ in range(math.ceil(wanted / MIN_YIELD)):
        if len(held) == wanted:
            return sorted(held)
        drawn = source.draw()
        if drawn is not None:
            held.add(drawn)
    return sorted(held) if len(held) == wanted else None
