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


def state_source(pst, resolver, dfa, leaf):
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
    source = StateSource(resolver, leaf, aim)
    return source if source.aims_land() else None


class StateSource:
    """Prefixes the tree places at one leaf."""

    def __init__(self, resolver, leaf, aim):
        self._population = resolver.population
        self._path = resolver.tree.path_of(leaf)
        # A split replaces a leaf with a node holding both ids, so every id the
        # tree reports has a path to it.
        assert self._path is not None, leaf
        self._aim = aim
        self._served = set()
        self._resting = []

    def draw(self, wanted: int) -> Optional[bytes]:
        """One string resting at the leaf, or ``None`` when there are no more.

        ``wanted`` is how many the caller is collecting, which is what sizes the
        read of what already rests there.

        Aims before serving for what it leaves behind rather than what it
        returns: a string pushed toward the leaf is one the population then
        holds, and the split test reads the population.
        """
        landed = self._aim_once()
        return self._resting_member(wanted) if landed is None else landed

    def aims_land(self) -> bool:
        """Whether aiming at this leaf lands often enough to keep asking.

        At yield ``MIN_YIELD`` one lands within ``1 / MIN_YIELD`` aims more often
        than not, so that many is what the leaf gets to prove itself in.  A floor
        on patience, like the yield itself, not a measurement of it.

        The draws are kept whichever way it goes: a string pushed toward the leaf
        is one the population then holds.
        """
        return any(
            self._aim_once() is not None for _ in range(math.ceil(1 / MIN_YIELD))
        )

    def _aim_once(self) -> Optional[bytes]:
        """An aimed string the tree rested here, or ``None`` where it rested it
        somewhere else."""
        aimed = self._aim()
        self._population.add(aimed)
        if self._population.settle(aimed, self._path):
            return aimed
        return None

    def _resting_member(self, wanted: int) -> Optional[bytes]:
        if not self._resting:
            # Reading a leaf pushes strings down to it, so the count is work
            # rather than a cap: ask for what could still be served, no more.
            self._resting = [
                m
                for m in self._population.members(
                    self._path, len(self._served) + wanted
                )
                if m not in self._served
            ]
        if not self._resting:
            return None
        member = self._resting.pop()
        self._served.add(member)
        return member


def collect(source, wanted: int) -> Optional[list]:
    """``wanted`` distinct prefixes from ``source``, or ``None`` if it could not.

    Giving up is the point: a population nothing can be drawn for is one the
    round cannot read a rate over, and saying so beats holding it to one.
    """
    held = set()
    for _ in range(math.ceil(wanted / MIN_YIELD)):
        if len(held) == wanted:
            return sorted(held)
        drawn = source.draw(wanted)
        if drawn is not None:
            held.add(drawn)
    return sorted(held) if len(held) == wanted else None
