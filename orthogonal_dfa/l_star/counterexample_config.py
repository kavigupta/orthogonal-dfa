"""``Level5Settings`` of ``OrthoDFA/Level5.lean``, and the one shipped."""

import math
from dataclasses import dataclass

#: Leaves past which a round gives up.
MAX_LEAVES = 100

#: Probes before a rate test may first decide, as the consistency estimate's.
FIRST_LOOK = 30


@dataclass(frozen=True)
class CounterexampleSearchConfig:
    """Only the values it does not derive are fields."""

    #: Chance a rate test errs at one look.
    level: float
    #: Probes a round draws before giving up.
    max_probes: int
    #: Each harvest test's threshold, as a multiple of its unit's undecided rate at a
    #: good state: ``acceptable_fnr`` per read.
    harvest_multiple: float
    #: Agreement the hypothesis is driven to.
    acc_threshold: float
    #: The share of prefixes the family may leave undecided.
    acceptable_fnr: float

    @property
    def k(self) -> int:
        return 0

    @property
    def max_leaves(self) -> int:
        return MAX_LEAVES

    @property
    def first_look(self) -> int:
        return FIRST_LOOK

    @property
    def disagreement_rate(self) -> float:
        return 1 - self.acc_threshold

    def start_rate(self, depth: int) -> float:
        """Per probe, whose start sift reads up to ``depth`` nodes."""
        return min(1.0, self.harvest_multiple * depth * self.acceptable_fnr)

    @property
    def edge_rate(self) -> float:
        return self.harvest_multiple * self.acceptable_fnr

    def middle_rate(self, queries_per_search: float) -> float:
        return min(
            1.0, self.harvest_multiple * queries_per_search * self.acceptable_fnr
        )

    @property
    def edge_excess(self) -> int:
        """An edge undecided at most ln 2 times ``edge_rate`` reaches it with
        chance under ``level``."""
        return math.ceil(math.log2(1 / self.level))

    @property
    def edge_traffic(self) -> float:
        return self.acceptable_fnr

    @property
    def search_share(self) -> float:
        return self.acceptable_fnr


def counterexample_search_config(
    acc_threshold: float, acceptable_fnr: float
) -> CounterexampleSearchConfig:
    return CounterexampleSearchConfig(
        level=1e-3,
        max_probes=50_000,
        harvest_multiple=5,
        acc_threshold=acc_threshold,
        acceptable_fnr=acceptable_fnr,
    )
