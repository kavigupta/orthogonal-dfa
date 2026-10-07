"""Where a boundary string came from, so that more can be drawn the same way.

A round's boundary strings are the reads its tree could not place.  Each read was
of a string drawn from some distribution -- a population's prefixes, a state
source's aims, the sampler's probes -- and made in one of two ways: sifted from
the root, extended or not by a letter, or met along a probe's walk.  A
provenance holds the distribution and the way, and ``sample`` draws afresh and
makes the same read through the same tree.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional

from .rejection_source import SourceDry
from .sifting import Sifter, anchored_walk, first_disagreeing_edge


@dataclass(frozen=True)
class Read:
    """What a read was of: a draw from ``distribution``, anything with a
    ``draw()``, extended by ``extension`` and sifted, or walked where
    ``extension`` is ``None``."""

    distribution: Any
    extension: Optional[bytes]

    def extended(self, extension: bytes) -> "Read":
        """The read of a draw ``extension`` further on; a walk stays a walk."""
        if self.extension is None:
            return self
        return Read(self.distribution, self.extension + extension)


class Aimed:
    """A state source's aim, as a distribution."""

    def __init__(self, aim):
        self._aim = aim

    def draw(self) -> bytes:
        return self._aim()


@dataclass(frozen=True, eq=False)
class Provenance(ABC):
    distribution: Any
    sifter: Sifter = field(repr=False)

    def sample(self) -> List[bytes]:
        """The strings the tree cannot place in a fresh draw read this way; none
        where the distribution has run dry."""
        try:
            drawn = self.distribution.draw()
        except SourceDry:
            return []
        return self._read(drawn)

    @abstractmethod
    def _read(self, drawn) -> List[bytes]: ...


@dataclass(frozen=True, eq=False)
class Sifted(Provenance):
    """A draw extended by ``extension`` and sifted from the root: a member pushed
    toward a leaf, or extended by a letter to vote on an edge."""

    extension: bytes

    def _read(self, drawn) -> List[bytes]:
        leaf, boundary = self.sifter.sift_and_boundary(drawn + self.extension)
        return [boundary] if leaf is None else []


@dataclass(frozen=True, eq=False)
class Walked(Provenance):
    """A probe walked as the counterexample pass walks one: anchored, sifted at
    its end, and searched for the disagreeing edge where walk and sift part."""

    transitions: dict = field(repr=False)

    def _read(self, drawn) -> List[bytes]:
        met = []

        def sift(seq):
            leaf, boundary = self.sifter.sift_and_boundary(seq)
            if leaf is None:
                met.append(boundary)
            return leaf

        start, states = anchored_walk(drawn, sift, self.transitions)
        if start is not None:
            landed = sift(drawn)
            if landed is not None and landed != states[-1]:
                first_disagreeing_edge(drawn, states, sift, start, len(drawn))
        return met


def provenance(read: Read, sifter, transitions) -> Provenance:
    """The provenance of a read the round made with ``sifter``, walking
    ``transitions``."""
    if read.extension is None:
        return Walked(read.distribution, sifter, transitions)
    return Sifted(read.distribution, sifter, read.extension)
