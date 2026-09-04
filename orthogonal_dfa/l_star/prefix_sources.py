"""Drawing the prefixes that belong to one state.

The hypothesis says where to aim a string; the tree says where it landed, and
only the tree's answer counts.
"""

import math
from typing import Optional

from .dfa_utils import (
    count_paths_to_state,
    sample_string_reaching_state,
    uniform_weights,
)

#: A source landing fewer of its aims than this is one to stop waiting for.  At
#: yield ``p`` it takes about ``1 / p`` draws per prefix, which is what bounds
#: the asking.  A floor on patience, not a measurement.
MIN_YIELD = 0.2


def _aim_at(pst, dfa, leaf):
    """Draw a string the hypothesis says reaches ``leaf``, or ``None`` where it
    says none does at the sampler's length."""
    weights = pst.sampler.symbol_weights(pst.alphabet_size)
    length = pst.sampler.length
    if not count_paths_to_state(dfa, leaf, length, uniform_weights(dfa))[length][
        dfa.initial_state
    ]:
        return None
    mass = count_paths_to_state(dfa, leaf, length, weights)
    return lambda: sample_string_reaching_state(dfa, mass, pst.rng, weights)


class StateSource:
    """Prefixes the tree places at one leaf."""

    def __init__(self, pst, resolver, dfa, leaf, *, serves):
        self._population = resolver.population
        self._path = resolver.tree.path_of(leaf)
        self._aim = _aim_at(pst, dfa, leaf)
        self._serves = serves
        self._served = set()
        self._resting = []

    def draw(self) -> Optional[bytes]:
        """One string resting at the leaf, or ``None`` when there are no more.

        Aims before serving for what it leaves behind rather than what it
        returns: a string pushed toward the leaf is one the population then
        holds, and the split test reads the population.
        """
        if self._path is None:
            return None
        if self._aim is not None:
            aimed = self._aim()
            if aimed is not None:
                self._population.add(aimed)
                if self._population.settle(aimed, self._path):
                    return aimed
        return self._resting_member()

    def _resting_member(self) -> Optional[bytes]:
        if not self._resting:
            # Reading a leaf pushes strings down to it, so the count is work
            # rather than a cap: ask for what could still be served, no more.
            self._resting = [
                m
                for m in self._population.members(
                    self._path, len(self._served) + self._serves
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
        drawn = source.draw()
        if drawn is not None:
            held.add(drawn)
    return sorted(held) if len(held) == wanted else None
