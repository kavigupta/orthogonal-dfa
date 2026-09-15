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
    """See the module docstring.  A subclass brings the attempted draw and the
    rates it is judged by: ``proving`` from `proving_attempts`, and ``poor``,
    below which `draw` gives up."""

    proving: tuple
    poor: float

    def __init__(self, served=()):
        # A draw landing on one of these is accepted all the same, so the rate
        # alone never says a source is spent.
        self._served = set(served)
        self._pool = []

    @abstractmethod
    def attempt_draw(self) -> bool:
        """Draw once, pool what lands in the region, and say whether it did."""

    @abstractmethod
    def source_repr(self) -> str:
        """Which source this is, for the error when it runs dry."""

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
        raise RuntimeError(
            f"Source {self.source_repr()} found no new samples in {dry} attempts"
        )
