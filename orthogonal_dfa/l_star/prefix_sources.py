"""Where each population of prefixes comes from.

A round ends by handing the next one a source per population rather than a list
of prefixes.  The next round draws what it needs when it needs it -- the gate
wanting more of one population to read a rate over costs a draw, not a redesign
-- and a source that cannot deliver is dropped along with the population it
feeds, which is the only honest thing to say about a state nothing reaches.

The sources close over the round that defined them: the tree that says where a
string rests, and the hypothesis that says where to aim one.
"""

import math
from collections import deque
from typing import Optional

from .dfa_utils import count_paths_to_state, sample_string_reaching_state
from .mask_table import UNIFORM

#: Prefixes a population is asked for.
WANTED = 100
#: A source landing fewer of its draws than this is one to stop waiting for.  At
#: yield ``p`` it takes about ``1 / p`` asks per prefix, which is what bounds the
#: asking.  A floor on patience, not a measurement.
MIN_YIELD = 0.2


class UniformSource:
    """The learner's own sampler.  Every draw is a prefix, so this never fails.

    Draws for the pool the table starts with and adds to, rather than for a
    population of its own: two populations of the same sampler would be one vote
    each however differently they had grown.
    """

    label = UNIFORM

    def __init__(self, pst):
        self._pst = pst

    def draw(self, wanted: int) -> Optional[bytes]:
        del wanted
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

    def draw(self, wanted: int) -> Optional[bytes]:
        """One unplaced string from a fresh probe, or ``None`` if it had none.

        A probe usually strands more than one, so the rest are kept for the next
        ask rather than resifted.
        """
        del wanted
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


def _aim_at(pst, dfa, leaf):
    """A draw of a string the hypothesis says reaches ``leaf``.

    It yields ``None`` where the sampler cannot make one of its length -- no
    path, or none its symbol weights would take -- which reads the same as a
    draw that missed: either way the leaf has only what already rests there.
    """
    weights = pst.sampler.symbol_weights(pst.alphabet_size)
    mass = count_paths_to_state(dfa, leaf, pst.sampler.length, weights)
    return lambda: sample_string_reaching_state(dfa, mass, pst.rng, weights)


class StateSource:
    """Prefixes the tree places at one leaf."""

    def __init__(self, pst, resolver, dfa, leaf, *, sink):
        self.label = ("state", leaf)
        self._population = resolver.population
        self._path = resolver.tree.path_of(leaf)
        # A split replaces a leaf with a node holding both ids, so every id the
        # tree reports has a path to it.
        assert self._path is not None, leaf
        self._aim = _aim_at(pst, dfa, leaf)
        self._sink = sink
        self._served = set()
        self._resting = []
        #: Draws a collection gave up on.  Aiming is expensive and the leaf is
        #: still where they rest, so they are the next ask's cheapest prefixes.
        self._spare = deque()

    def draw(self, wanted: int) -> Optional[bytes]:
        """One string resting at the leaf, or ``None`` when there are no more.

        ``wanted`` is how many the caller is collecting, which is what sizes the
        read of what already rests there.

        Aims before serving for what it leaves behind rather than what it
        returns: a string pushed toward the leaf is one the population then
        holds, and the split test reads the population.
        """
        if self._spare:
            return self._spare.popleft()
        aimed = self._aim()
        if aimed is not None:
            self._population.add(aimed)
            # Where it rests, not where it was aimed.
            if self._population.settle(aimed, self._path):
                return aimed
            if self._population.resting_at(aimed) is None:
                self._sink(aimed)
        return self._resting_member(wanted)

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

    def unused(self, drawn) -> None:
        """Take back draws that did not become a population."""
        self._spare.extend(drawn)


def gather(source, wanted: int) -> list:
    """Up to ``wanted`` distinct prefixes from ``source``, however few it gives.

    For growing a population, where any is a gain.  Defining one is ``collect``,
    which holds out for the whole number.
    """
    held = set()
    for _ in range(math.ceil(wanted / MIN_YIELD)):
        if len(held) == wanted:
            break
        drawn = source.draw(wanted)
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
    # Taking is only earned by a population coming of it.  A source that holds
    # nothing has nothing to take back; one that buffers a finite supply would
    # otherwise be emptied by every ask it could not meet.
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
