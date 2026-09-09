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


def state_source(pst, resolver, dfa, leaf, *, sink):
    """A source for ``leaf``, or ``None`` where aiming at it does not land.

    Two ways it does not.  The hypothesis may have no string of the sampler's
    length reaching the leaf at all, and then there is nothing to aim.  Or it may
    have plenty and the tree place almost none of them there -- the hypothesis
    says where to aim, the tree says where it lands, and they disagree.

    Either way what is left is whatever already rests at the leaf, which runs
    out.  That is a finite population wearing an infinite one's clothes, and the
    round would come up short on it forever, so it is probed before it is kept.
    """
    aim = _aim_at(pst, dfa, leaf)
    if aim is None:
        return None
    source = StateSource(resolver, leaf, aim, sink=sink)
    return source if source.aims_land() else None


class StateSource:
    """Prefixes the tree places at one leaf."""

    def __init__(self, resolver, leaf, aim, *, sink):
        self.label = ("state", leaf)
        self._population = resolver.population
        self._path = resolver.tree.path_of(leaf)
        # A split replaces a leaf with a node holding both ids, so every id the
        # tree reports has a path to it.
        assert self._path is not None, leaf
        self._aim = aim
        self._sink = sink
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
        if self._population.resting_at(aimed) is None:
            self._sink(aimed)
        return False

    def aims_land(self) -> bool:
        """Whether aiming at this leaf lands often enough to keep asking.

        At yield ``MIN_YIELD`` one lands within ``1 / MIN_YIELD`` aims more often
        than not, so that many is what the leaf gets to prove itself in.  A floor
        on patience, like the yield itself, not a measurement of it.
        """
        return any(self.aimed_draw() for _ in range(math.ceil(1 / MIN_YIELD)))

    def draw(self, wanted: int) -> Optional[bytes]:
        """One prefix resting at the leaf, or ``None`` once it has no more.

        What the pool does not hold it aims for, so running out means the leaf's
        whole reachable support has been served -- not that a draw missed.

        ``wanted`` sizes the one read of the leaf.  Reading pushes strings down
        to it, so the count is work rather than a cap: ask for what could still
        be served, no more.
        """
        while True:
            while self._pool:
                member = self._pool.pop()
                if member not in self._served:
                    self._served.add(member)
                    return member
            if not self._read_the_leaf:
                self._read_the_leaf = True
                self._pool.extend(
                    self._population.members(self._path, len(self._served) + wanted)
                )
                continue
            # Aims that only bring back what has been served already are the
            # leaf saying it has nothing else, which is different from missing.
            if not any(self.aimed_draw() for _ in range(math.ceil(1 / MIN_YIELD))):
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
