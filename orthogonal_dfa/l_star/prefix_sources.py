"""Drawing the prefixes that belong to one state.

A source belongs to the round that made it: it closes over that round's tree,
which says where a string rests, and its hypothesis, which says where to aim
one.  Only the tree's answer counts.
"""

from math import isqrt

from .dfa_utils import (
    count_paths_to_state,
    sample_string_reaching_state,
    uniform_weights,
)
from .rejection_source import RejectionSource, proving_attempts

#: A leaf landing at least this share of its aims is one worth asking again.
GOOD_YIELD = 0.5
#: One landing at most this share is one to stop asking.  Nothing decides
#: anything between the two bars, which is what keeps the count that tells them
#: apart affordable.
POOR_YIELD = 0.25


class UniformSource:
    """The learner's own sampler.  Every draw is a prefix, so this never fails."""

    def __init__(self, pst):
        self._pst = pst

    def draw(self) -> bytes:
        return self._pst.sampler.sample(
            self._pst.rng, alphabet_size=self._pst.alphabet_size
        )

    @property
    def proven(self) -> bool:
        return True


class MidfixSource(UniformSource):
    """The sampler's draws cut to their first ``length`` symbols and followed by
    ``midfix``."""

    def __init__(self, pst, length, midfix):
        super().__init__(pst)
        self._length = length
        self._midfix = midfix

    def draw(self) -> bytes:
        return super().draw()[: self._length] + self._midfix

    def worth_drawing(self) -> bool:
        return True


class HarvestSource(RejectionSource):
    """More of a population a refusal held: each attempt reads a fresh draw the
    way the refusal sample read its draws."""

    def __init__(self, read, *, known, acc_threshold):
        """``read()`` reads a fresh draw and returns what it leaves.  Worth
        drawing on where an attempt turns up a new string at least
        ``1 - acc_threshold`` of the time, and not at half that."""
        super().__init__()
        assert acc_threshold < 1, acc_threshold
        self._good = 1 - acc_threshold
        self._served.update(known)
        self._seen = set(known)
        self._read = read

    def attempt_draw(self) -> bool:
        fresh = [string for string in self._read() if string not in self._seen]
        self._seen.update(fresh)
        self._pool.extend(fresh)
        return bool(fresh)

    @property
    def proving(self) -> tuple:
        return proving_attempts(self._good, self._good / 2)

    @property
    def poor(self) -> float:
        return self._good / 2

    def source_repr(self) -> str:
        return "harvest"


def aim_at(pst, dfa, leaf):
    """
    Attempt to aim at a leaf, returning a source that draws on it or ``None`` if the leaf is
    unreachable from the starting state, or if too few strings of the sampler's
    length reach it to draw from without redrawing what it has already drawn.

    Aims are drawn with replacement, so a leaf that holds

        sqrt(alphabet_size ** length)

    of the strings of that length repeats one only after the fourth root of
    them have been drawn.  Scaled to the space rather than to the draws asked
    for: a leaf is thin compared to what the sampler could have put there.
    """
    weights = pst.sampler.symbol_weights(pst.alphabet_size)
    length = pst.sampler.length
    # Counted evenly rather than off the mass below: the sampler's weights are
    # read as ratios, so its mass is on no scale a threshold could name.
    reaching = count_paths_to_state(dfa, leaf, length, uniform_weights(dfa))
    if reaching[length][dfa.initial_state] < isqrt(pst.alphabet_size**length):
        return None
    mass = count_paths_to_state(dfa, leaf, length, weights)
    # A symbol the sampler never places can leave a well-reached leaf with none.
    if mass[length][dfa.initial_state] == 0:
        return None
    return lambda: sample_string_reaching_state(dfa, mass, pst.rng, weights)


def state_source(resolver, leaf, aim, *, wanted):
    """
    A source that draws on `aim` and guarantees (up to the misread chance in
    `proving_attempts`) that at least POOR_YIELD (25%) of the strings it draws
    will land at the given leaf, according to the tree in `resolver`.

    If this guarantee cannot be made, returns None
    """
    source = StateSource(resolver, leaf, aim, wanted=wanted)
    return source if source.worth_drawing() else None


class StateSource(RejectionSource):
    """Prefixes the tree places at one leaf."""

    proving = proving_attempts(GOOD_YIELD, POOR_YIELD)
    poor = POOR_YIELD

    def __init__(self, resolver, leaf, aim, *, wanted):
        super().__init__()
        self._population = resolver.population
        self._path = resolver.tree.path_of(leaf)
        # A split replaces a leaf with a node holding both ids, so every id the
        # tree reports has a path to it.
        assert self._path is not None, leaf
        self._aim = aim
        #: Resting at the leaf and not yet handed out.  Reading a leaf pushes
        #: strings down to it, so the count is work rather than a cap: ask for
        #: what this source is being built to serve.
        self._pool.extend(self._population.members(self._path, wanted))

    def attempt_draw(self) -> bool:
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

    def source_repr(self) -> str:
        return f"leaf {self._path}"


def draw_many(source, wanted: int) -> list:
    """``wanted`` prefixes from ``source``, sorted rather than in draw order."""
    return sorted(source.draw() for _ in range(wanted))
