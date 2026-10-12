"""``TallyCfg`` of ``OrthoDFA/TallyLoop.lean``, and the one the learner ships."""

import math
from dataclasses import dataclass

#: Leaves past which a round gives up.
MAX_LEAVES = 100

#: Probes before a rate test may first decide, as the consistency estimate's.
FIRST_LOOK = 30


@dataclass(frozen=True)
class TallyConfig:
    """``TallyCfg``, from the values it does not derive."""

    #: Records at one target before the edge they are at is fixed.
    m: int
    #: Chance a rate test errs at one look.
    level: float
    #: Probes a round draws before giving up.
    max_probes: int
    #: Each harvest's undecided rate, as a multiple of ``acceptable_fnr``.
    start_multiple: float
    edge_multiple: float
    middle_multiple: float
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

    @property
    def start_rate(self) -> float:
        return self.start_multiple * self.acceptable_fnr

    @property
    def edge_rate(self) -> float:
        return self.edge_multiple * self.acceptable_fnr

    @property
    def middle_rate(self) -> float:
        return self.middle_multiple * self.acceptable_fnr

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


def tally_config(acc_threshold: float, acceptable_fnr: float) -> TallyConfig:
    return TallyConfig(
        m=5,
        level=1e-3,
        max_probes=50_000,
        start_multiple=5,
        edge_multiple=10,
        middle_multiple=20,
        acc_threshold=acc_threshold,
        acceptable_fnr=acceptable_fnr,
    )
