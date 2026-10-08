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
from .sifting import Sifter, check_from, read_from, walk_from


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
class _FromStart(Provenance):
    """A probe read as the round's pass reads one: from where the cut places its
    first ``k`` symbols, along the learned ``transitions``."""

    transitions: dict = field(repr=False)
    k: int


@dataclass(frozen=True, eq=False)
class Walked(_FromStart):
    """Every read the counterexample check makes of a probe that the cut cannot
    place."""

    def _middle(self, seq):
        return self.sifter.halfway(seq)[0]

    def _read(self, drawn) -> List[bytes]:
        met = []

        def sift(seq):
            leaf, boundary = self.sifter.sift_and_boundary(seq)
            if leaf is None:
                met.append(boundary)
            return leaf, boundary

        check_from(drawn, sift, self._middle, self.transitions, self.k)
        return met


@dataclass(frozen=True, eq=False)
class WalkBlocked(_FromStart):
    """What the walk of a probe leaves where it is blocked."""

    def _read(self, drawn) -> List[bytes]:
        _, block = walk_from(
            drawn, self.sifter.sift_and_boundary, self.transitions, self.k
        )
        return _found(block)


@dataclass(frozen=True, eq=False)
class ReadBlocked(_FromStart):
    """What the walk and the sift of a probe leave where they are blocked."""

    def _read(self, drawn) -> List[bytes]:
        _, block, _ = read_from(
            drawn, self.sifter.sift_and_boundary, self.transitions, self.k
        )
        return _found(block)


def _found(block) -> List[bytes]:
    return [] if block is None or block.found is None else [block.found]


def provenance(read: Read, sifter, transitions, k) -> Provenance:
    """The provenance of a read the round made with ``sifter``, its pass walking
    the learned ``transitions`` from ``k``."""
    if read.extension is None:
        return Walked(read.distribution, sifter, transitions, k)
    return Sifted(read.distribution, sifter, read.extension)
