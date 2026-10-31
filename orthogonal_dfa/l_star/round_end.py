"""How a counterexample search ends: its per-site counts, and the tests on them."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .counterexample_config import CounterexampleSearchConfig
from .counterexample_probe import Charges, Middle, Outcome
from .hypothesis_dfa import Edge
from .statistics import seq_rate

SUCCESS = "success"
HARVEST_ENDS = "harvest ends"
HARVEST_STUCK = "harvest stuck"
HARVEST_MIDDLES = "harvest middles"
OUT_OF_PROBES = "out of probes"
HARVESTS = (HARVEST_ENDS, HARVEST_STUCK, HARVEST_MIDDLES)


@dataclass
class Ending:
    kind: str
    harvest: List[bytes]


@dataclass
class SiteCounts:
    """Since the hypothesis last changed."""

    probes: int = 0
    ends: List[bytes] = field(default_factory=list)
    stuck: List[bytes] = field(default_factory=list)
    edge_harvest: Dict[Edge, List[bytes]] = field(default_factory=dict)
    searches: int = 0
    middles: List[bytes] = field(default_factory=list)
    search_queries: int = 0

    def add(self, charged: Charges, outcome: Outcome):
        self.probes += 1
        self.ends += charged.ends
        self.stuck += charged.stuck
        self.middles += charged.middle
        self.search_queries += charged.search_queries
        self.searches += isinstance(outcome, Middle)
        for edge, strings in charged.undecided_by_edge.items():
            self.edge_harvest.setdefault(edge, []).extend(strings)


def ending_test(
    counts: SiteCounts,
    config: CounterexampleSearchConfig,
    depth: int,
) -> Optional[Ending]:
    """The harvest tests, then the success test."""
    if seq_rate(
        config.ends_rate(depth),
        config.ends_alternative(depth),
        config.level,
        counts.probes,
        len(counts.ends),
    ):
        return Ending(HARVEST_ENDS, list(counts.ends))
    if seq_rate(
        config.stuck_rate,
        config.stuck_alternative,
        config.level,
        counts.probes,
        len(counts.stuck),
    ):
        return Ending(HARVEST_STUCK, list(counts.stuck))
    searches = counts.searches
    per_search = counts.search_queries / max(searches, 1)
    if config.search_share * counts.probes <= searches and seq_rate(
        config.middle_rate(per_search),
        config.middle_alternative(per_search),
        config.level,
        searches,
        len(counts.middles),
    ):
        return Ending(HARVEST_MIDDLES, list(counts.middles))
    if succeeds_at(counts, config, counts.probes):
        return Ending(SUCCESS, [])
    return None


def succeeds_at(
    counts: SiteCounts, config: CounterexampleSearchConfig, probes: int
) -> bool:
    """The success test, had the hypothesis held for ``probes`` probes."""
    return seq_rate(
        config.disagreement_rate,
        config.disagreement_alternative,
        config.level,
        probes,
        counts.searches + len(counts.stuck),
    )
