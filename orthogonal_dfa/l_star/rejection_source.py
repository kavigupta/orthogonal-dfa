"""Drawing from a region of a space by drawing from the space and keeping what
lands in it."""

from abc import ABC, abstractmethod
from math import ceil, log

import scipy.stats

from .statistics import binom_cdf

#: Chance of reading an acceptance rate as either bar when it is the other.
_MISREAD = 1e-5


def proving_attempts(good, poor):
    """Sizes the test so that P(reject | rate >= ``good``) and
    P(keep | rate <= ``poor``) are both bounded by ``_MISREAD``."""
    attempts = 0
    while True:
        attempts += 1
        accepted = int(scipy.stats.binom.isf(_MISREAD, attempts, poor))
        if binom_cdf(accepted, attempts, good) <= _MISREAD:
            return attempts, accepted


class RejectionSource(ABC):
    """See the module docstring."""

    def __init__(self):
        self._served = set()
        self._pool = []

    @property
    @abstractmethod
    def proving(self) -> tuple:
        """(attempts, accepted) from `proving_attempts`."""

    @property
    @abstractmethod
    def poor(self) -> float:
        """The acceptance rate below which `draw` calls a source dry."""

    @abstractmethod
    def attempt_draw(self) -> bool:
        """Draw once, pool what lands in the region, and say whether it did."""

    @abstractmethod
    def source_repr(self) -> str:
        """Which source this is, for the error when it runs dry."""

    def found(self) -> list:
        """What has been accepted and not yet served, including during
        `has_sufficient_yield`."""
        got = [member for member in self._pool if member not in self._served]
        self._served.update(got)
        self._pool.clear()
        return got

    def has_sufficient_yield(self) -> bool:
        attempts, accepted = self.proving
        return sum(self.attempt_draw() for _ in range(attempts)) > accepted

    def draw(self, false_alarm_p=1e-9) -> bytes:
        """One string from the pool, drawing for more when it runs dry."""
        # Attempts in a row accepting nothing new before a source is called dry.
        # Geometric distribution.
        dry = ceil(log(false_alarm_p) / log(1 - self.poor))
        # One pass more than that: what an attempt pooled is read by the drain
        # of the pass after it.
        for _ in range(dry + 1):
            while self._pool:
                member = self._pool.pop()
                if member not in self._served:
                    self._served.add(member)
                    return member
            self.attempt_draw()
        # A draw landing on a string already served is accepted all the same, so
        # the rate alone never says a source is spent.
        raise RuntimeError(
            f"Source {self.source_repr()} found no new samples in {dry} attempts"
        )
