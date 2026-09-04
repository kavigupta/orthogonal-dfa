"""Where each population of prefixes comes from.

A round ends by handing the next one a source per population rather than a list
of prefixes.  The next round draws what it needs when it needs it -- the gate
wanting more of one population to read a rate over costs a draw, not a redesign
-- and a source that cannot deliver is dropped along with the population it
feeds, which is the only honest thing to say about a state nothing reaches.

The sources close over the round that defined them: the tree that says where a
string rests, and the hypothesis that says where to aim one.  What a source
cannot place is not thrown away, since a string no family could place is what
the indecisive source serves.
"""

import math
from collections import deque
from typing import Optional

from .dfa_utils import (
    count_paths_to_state,
    sample_string_reaching_state,
    uniform_weights,
)

#: A source landing fewer of its aims than this is one to stop waiting for.  At
#: yield ``p`` it takes about ``1 / p`` draws per prefix, which is what bounds
#: the asking.  A floor on patience rather than a measurement: the rate a source
#: actually manages is what ``collect`` finds out.
MIN_YIELD = 0.2


class StateSource:
    """Prefixes the tree places at one leaf.

    The hypothesis says where to aim; the tree says where the string went.  Only
    the tree's answer counts, so a draw the tree places elsewhere -- or cannot
    place at all -- is not a prefix for this population.

    What the population already rests at the leaf comes first.  Those are the
    same answer from the same arbiter, already paid for, and a leaf holding
    hundreds of them is not one to go aiming at: aiming is how a leaf nothing has
    reached yet gets its first prefixes, not how a leaf gets every prefix.
    """

    def __init__(self, pst, resolver, dfa, leaf, *, serves, sink=None):
        self.label = ("state", leaf)
        self._resting = []
        self._serves = serves
        self._served = set()
        #: Draws a collection gave up on.  Aiming is expensive and the leaf is
        #: still where they rest, so they are the next ask's cheapest prefixes.
        self._spare = deque()
        self._pst = pst
        self._resolver = resolver
        self._dfa = dfa
        self._leaf = leaf
        self._sink = sink
        self._path = resolver.tree.path_of(leaf)
        weights = pst.sampler.symbol_weights(pst.alphabet_size)
        length = pst.sampler.length
        counts = count_paths_to_state(dfa, leaf, length, uniform_weights(dfa))
        self._reachable = counts[length][dfa.initial_state]
        self._mass = (
            count_paths_to_state(dfa, leaf, length, weights)
            if self._reachable
            else None
        )
        self._weights = weights

    def draw(self) -> Optional[bytes]:
        """One aimed string, or one already resting at the leaf if that misses.

        Aiming comes first for what it leaves behind rather than what it returns:
        a string pushed toward the leaf is a string the population then holds,
        and the split test reads the population.  Serve a resting member in its
        place and this population fills while the tree gains nothing to split.
        """
        if self._path is None:
            return None
        if self._spare:
            return self._spare.popleft()
        population = self._resolver.population
        if self._reachable:
            aimed = sample_string_reaching_state(
                self._dfa, self._mass, self._pst.rng, self._weights
            )
            if aimed is not None:
                population.add(aimed)
                # Where it rests, not where it was aimed.
                if population.settle(aimed, self._path):
                    return aimed
                if self._sink is not None and population.resting_at(aimed) is None:
                    self._sink(aimed)
        # Aiming misses most of the time, and the leaf's own members are what
        # this population is made of anyway.
        if not self._resting:
            # As many as it could still be asked for, no deeper: reading a leaf
            # pushes strings down to it, so the count is work and not just a cap.
            self._resting = [
                m
                for m in population.members(
                    self._path, len(self._served) + self._serves
                )
                if m not in self._served
            ]
        if self._resting:
            resting = self._resting.pop()
            self._served.add(resting)
            return resting
        return None

    def unused(self, drawn) -> None:
        """Take back draws that did not become a population."""
        self._spare.extend(drawn)


def collect(source, wanted: int):
    """``wanted`` prefixes from ``source``, or ``None`` if it could not.

    Giving up is the point: a population nothing can be drawn for is one the
    round cannot read a rate over, and saying so beats holding it to one.
    """
    held, budget = [], math.ceil(wanted / MIN_YIELD)
    seen = set()
    while len(held) < wanted and budget:
        budget -= 1
        drawn = source.draw()
        if drawn is not None and drawn not in seen:
            seen.add(drawn)
            held.append(drawn)
    if len(held) >= wanted:
        return held
    # Taking is only earned by a population coming of it.  A source that holds
    # nothing has nothing to take back; one that buffers a finite supply would
    # otherwise be emptied by every ask it could not meet.
    give_back = getattr(source, "unused", None)
    if give_back is not None:
        give_back(held)
    return None
