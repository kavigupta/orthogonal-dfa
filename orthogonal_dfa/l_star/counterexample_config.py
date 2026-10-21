"""``Level5Settings`` of ``OrthoDFA/Level5.lean``, and the one shipped."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CounterexampleSearchConfig:
    """Only the values it does not derive are fields."""

    #: Chance a rate test ever errs.
    level: float
    #: Probes a round draws before giving up.
    max_probes: int
    #: Each harvest test's threshold, as a multiple of its unit's undecided rate at a
    #: good state: ``acceptable_fnr`` per read.
    harvest_multiple: float
    #: The stuck test's rate, as a multiple of ``disagreement_rate``.
    stuck_ratio: float
    #: Each sequential test's alternative, as a multiple of its rate.
    ends_alternative_ratio: float
    stuck_alternative_ratio: float
    middle_alternative_ratio: float
    disagreement_alternative_ratio: float
    #: Agreement the hypothesis is driven to.
    acc_threshold: float
    #: The share of prefixes the family may leave undecided.
    acceptable_fnr: float
    #: Where the probes' walks start.
    k: int = 0

    @property
    def disagreement_rate(self) -> float:
        return 1 - self.acc_threshold

    def ends_rate(self, depth: int) -> float:
        """Per probe, whose start and end sifts read up to ``depth`` nodes each;
        at most a half, so its alternative is a rate."""
        return min(0.5, self.harvest_multiple * 2 * depth * self.acceptable_fnr)

    def middle_rate(self, queries_per_search: float) -> float:
        return min(
            1.0, self.harvest_multiple * queries_per_search * self.acceptable_fnr
        )

    def ends_alternative(self, depth: int) -> float:
        return min(1.0, self.ends_alternative_ratio * self.ends_rate(depth))

    def middle_alternative(self, queries_per_search: float) -> float:
        return min(
            1.0, self.middle_alternative_ratio * self.middle_rate(queries_per_search)
        )

    @property
    def stuck_rate(self) -> float:
        return self.stuck_ratio * self.disagreement_rate

    @property
    def stuck_alternative(self) -> float:
        return self.stuck_alternative_ratio * self.stuck_rate

    @property
    def disagreement_alternative(self) -> float:
        return self.disagreement_alternative_ratio * self.disagreement_rate

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
        stuck_ratio=0.1,
        ends_alternative_ratio=2,
        stuck_alternative_ratio=2,
        middle_alternative_ratio=1.25,
        disagreement_alternative_ratio=0.1,
        acc_threshold=acc_threshold,
        acceptable_fnr=acceptable_fnr,
    )
