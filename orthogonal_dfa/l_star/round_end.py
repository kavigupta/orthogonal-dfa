"""How a counterexample search ends: its per-site counts, and the tests on them."""

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .counterexample_config import CounterexampleSearchConfig
from .counterexample_probe import Charges, Middle, Outcome
from .hypothesis_dfa import Edge, Path
from .statistics import seq_rate

SUCCESS = "success"
HARVEST_START = "harvest start"
HARVEST_EDGE = "harvest edge"
HARVEST_MIDDLES = "harvest middles"
OUT_OF_PROBES = "out of probes"
HARVESTS = (HARVEST_START, HARVEST_EDGE, HARVEST_MIDDLES)


@dataclass
class Ending:
    kind: str
    harvest: List[bytes]
    edge: Optional[Edge]


@dataclass
class SiteCounts:
    """Since the hypothesis last changed."""

    probes: int = 0
    start_harvest: List[bytes] = field(default_factory=list)
    traversals: Counter = field(default_factory=Counter)
    edge_harvest: Dict[Edge, List[bytes]] = field(default_factory=dict)
    searches: int = 0
    middles: List[bytes] = field(default_factory=list)
    search_queries: int = 0

    def add(self, charged: Charges, outcome: Outcome):
        self.probes += 1
        self.start_harvest += charged.start
        self.middles += charged.middle
        self.search_queries += charged.search_queries
        self.searches += isinstance(outcome, Middle)
        self.traversals.update(charged.traversals)
        for edge, strings in charged.undecided_by_edge.items():
            self.edge_harvest.setdefault(edge, []).extend(strings)


def ending_test(
    counts: SiteCounts,
    config: CounterexampleSearchConfig,
    depth: int,
    leaves: Set[Path],
) -> Optional[Ending]:
    """The harvest tests, then the success test."""
    if seq_rate(
        config.start_rate(depth),
        config.start_alternative(depth),
        config.level,
        counts.probes,
        len(counts.start_harvest),
    ):
        return Ending(HARVEST_START, list(counts.start_harvest), None)
    for edge, count in counts.traversals.items():
        harvest = counts.edge_harvest.get(edge, [])
        if (
            edge[0] in leaves
            and config.edge_traffic * counts.probes <= count
            and config.edge_excess <= len(harvest) - config.edge_rate * count
        ):
            return Ending(HARVEST_EDGE, list(harvest), edge)
    searches = counts.searches
    per_search = counts.search_queries / max(searches, 1)
    if config.search_share * counts.probes <= searches and seq_rate(
        config.middle_rate(per_search),
        config.middle_alternative(per_search),
        config.level,
        searches,
        len(counts.middles),
    ):
        return Ending(HARVEST_MIDDLES, list(counts.middles), None)
    if succeeds_at(counts, config, counts.probes):
        return Ending(SUCCESS, [], None)
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
        counts.searches,
    )
