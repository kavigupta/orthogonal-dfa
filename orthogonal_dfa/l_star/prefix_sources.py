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
from .mask_table import UNIFORM
from .rejection_source import RejectionSource, proving_attempts
from .sifting import PROBE_BLOCK, anchored_walk, first_disagreeing_edge

#: Prefixes a population is asked for.
WANTED = 100
#: A leaf landing at least this share of its aims is one worth asking again.
GOOD_YIELD = 0.5
#: One landing at most this share is one to stop asking.  Nothing decides
#: anything between the two bars, which is what keeps the count that tells them
#: apart affordable.
POOR_YIELD = 0.25

#: A probe turning up a boundary string at least this often is worth drawing on.
GOOD_BOUNDARY_YIELD = 0.2
#: One turning up at most this often is not.
POOR_BOUNDARY_YIELD = 0.1


class UniformSource:
    """The learner's own sampler.  Every draw is a prefix, so this never fails."""

    label = UNIFORM

    def __init__(self, pst):
        self._pst = pst

    def draw(self) -> bytes:
        return self._pst.sampler.sample(
            self._pst.rng, alphabet_size=self._pst.alphabet_size
        )


class BoundarySource(RejectionSource):
    """Strings the round's tree cannot place, asked about along a probe's walk.

    Only prefixes at least half the sampler's length are kept.  There are at least

        sqrt(alphabet_size ** length)

    of those, so a family straddling any share worth drawing on has more than a
    round can exhaust; the short prefixes are asked about by every probe and run
    out at once.
    """

    proving = proving_attempts(GOOD_BOUNDARY_YIELD, POOR_BOUNDARY_YIELD)
    poor = POOR_BOUNDARY_YIELD

    def __init__(self, pst, sifter, transitions, *, label, known):
        super().__init__()
        self.label = label
        self._served.update(known)
        self._pst = pst
        self._sifter = sifter
        self._transitions = transitions
        self._long_enough = -(-pst.sampler.length // 2)
        self._seen = set(known)
        self._probes = []

    def _sift(self, seq):
        leaf, boundary = self._sifter.sift_and_boundary(seq)
        if (
            leaf is None
            and len(seq) >= self._long_enough
            and boundary not in self._seen
        ):
            self._seen.add(boundary)
            self._pool.append(boundary)
        return leaf

    def attempt_draw(self) -> bool:
        before = len(self._seen)
        if not self._probes:
            self._probes = [
                self._pst.sampler.sample(
                    self._pst.rng, alphabet_size=self._pst.alphabet_size
                )
                for _ in range(PROBE_BLOCK)
            ]
            self._sifter.prefill(self._probes)
        probe = self._probes.pop()
        start, states = anchored_walk(probe, self._sift, self._transitions)
        if start is not None:
            landed = self._sift(probe)
            if landed is not None and landed != states[-1]:
                # Called for the sifts it makes on the way; the edge it returns
                # is the round's, not this source's.
                first_disagreeing_edge(probe, states, self._sift, start, len(probe))
        return len(self._seen) > before

    def source_repr(self) -> str:
        return "boundary"


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
    return source if source.has_sufficient_yield() else None


class StateSource(RejectionSource):
    """Prefixes the tree places at one leaf."""

    proving = proving_attempts(GOOD_YIELD, POOR_YIELD)
    poor = POOR_YIELD

    def __init__(self, resolver, leaf, aim, *, wanted):
        self.label = ("state", leaf)
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
    """``wanted`` prefixes from ``source``.

    A source proven worth drawing on has more than a round can use up -- the
    space it draws from is at least the square root of the whole -- so this asks
    for the number it wants rather than settling for what comes.
    """
    return sorted(source.draw() for _ in range(wanted))
