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

from typing import Optional

from .dfa_utils import (
    count_paths_to_state,
    sample_string_reaching_state,
    uniform_weights,
)

#: Prefixes a population is asked for.
WANTED = 100
#: Members of a leaf read back before a source starts aiming.  Loose: what it
#: bounds is the cost of asking, and a leaf with more than this to offer is not
#: one that needs aiming at all.
_RESTING_LIMIT = 2000
#: Draws allowed per prefix wanted before a source is given up on.  One that
#: survives yields at least a fifth of what it draws, so asking it for more later
#: costs about what asking it for these did.
ATTEMPTS_PER_PREFIX = 5


class UniformSource:
    """The learner's own sampler.  Every draw is a prefix, so this never fails."""

    label = "baseline"

    def __init__(self, pst):
        self._pst = pst

    def draw(self) -> Optional[bytes]:
        return self._pst.sampler.sample(
            self._pst.rng, alphabet_size=self._pst.alphabet_size
        )


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

    def __init__(self, pst, resolver, dfa, leaf, *, sink=None):
        self.label = ("state", leaf)
        self._resting = None
        self._served = set()
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
        if self._resting is None:
            self._resting = list(population.members(self._path, _RESTING_LIMIT))
        while self._resting:
            resting = self._resting.pop()
            if resting not in self._served:
                self._served.add(resting)
                return resting
        return None


class IndecisiveSource:
    """Strings no family could place, served from what the round already found.

    Kept rather than redrawn: they are the by-product of every other source's
    validation, and a string that straddles is expensive to find on purpose and
    free to notice in passing.
    """

    label = "boundary"

    def __init__(self, reservoir=(), refill=None):
        self._held = list(reservoir)
        self._served = set()
        self._refill = refill

    def offer(self, string) -> None:
        if string not in self._served:
            self._held.append(string)

    def draw(self) -> Optional[bytes]:
        while self._held:
            string = self._held.pop()
            if string not in self._served:
                self._served.add(string)
                return string
        return self._refill() if self._refill is not None else None


def collect(source, wanted: int = WANTED, attempts_per: int = ATTEMPTS_PER_PREFIX):
    """``wanted`` prefixes from ``source``, or ``None`` if it could not.

    Giving up is the point: a population nothing can be drawn for is one the
    round cannot read a rate over, and saying so beats holding it to one.
    """
    held, budget = [], wanted * attempts_per
    seen = set()
    while len(held) < wanted and budget:
        budget -= 1
        drawn = source.draw()
        if drawn is not None and drawn not in seen:
            seen.add(drawn)
            held.append(drawn)
    return held if len(held) >= wanted else None


def draw_for_split(source, wanted: int):
    """Prefixes from ``source`` to read the split on and nothing else.

    A population that cannot certify on the prefixes it holds needs more of its
    own, not more uniform ones -- and read only for the split, since adding them
    to the table costs a query on every fully observed column and unsettles the
    FNR the round has just met.
    """
    return collect(source, wanted=wanted) or []
