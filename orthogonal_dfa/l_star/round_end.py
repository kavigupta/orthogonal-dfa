"""How a counterexample search ends: its per-site counts, and the tests on them."""

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .counterexample_config import CounterexampleSearchConfig
from .counterexample_probe import Charges, EdgeFound, Middle, Outcome
from .hypothesis_dfa import Edge, Path
from .statistics import rate_side

SUCCESS = "success"
HARVEST_START = "harvest start"
HARVEST_EDGE = "harvest edge"
HARVEST_MIDDLES = "harvest middles"
TOO_BIG = "too big"
OUT_OF_PROBES = "out of probes"
HARVESTS = (HARVEST_START, HARVEST_EDGE, HARVEST_MIDDLES)


@dataclass
class Ending:
    kind: str
    harvest: List[bytes]
    edge: Optional[Edge]


@dataclass
class SiteCounts:
    """Over the round, and over the stretch since the hypothesis last changed."""

    probes: int = 0
    start_harvest: List[bytes] = field(default_factory=list)
    traversals: Counter = field(default_factory=Counter)
    edge_harvest: Dict[Edge, List[bytes]] = field(default_factory=dict)
    stretch_probes: int = 0
    stretch_searches: int = 0
    stretch_middles: List[bytes] = field(default_factory=list)
    stretch_search_queries: int = 0

    def add(self, charged: Charges, outcome: Outcome):
        self.probes += 1
        self.stretch_probes += 1
        self.start_harvest += charged.start
        self.stretch_middles += charged.middle
        self.stretch_search_queries += charged.search_queries
        self.stretch_searches += isinstance(outcome, (EdgeFound, Middle))
        self.traversals.update(charged.traversals)
        for edge, strings in charged.undecided_by_edge.items():
            self.edge_harvest.setdefault(edge, []).extend(strings)

    def reset_stretch(self):
        self.stretch_probes = 0
        self.stretch_searches = 0
        self.stretch_middles = []
        self.stretch_search_queries = 0


def ending_test(
    counts: SiteCounts,
    config: CounterexampleSearchConfig,
    depth: int,
    leaves: Set[Path],
) -> Optional[Ending]:
    """The harvest tests, then the success test."""
    start_rate = config.start_rate(depth)
    if _rate_above(config, start_rate, counts.probes, len(counts.start_harvest)):
        return Ending(HARVEST_START, list(counts.start_harvest), None)
    for edge, count in counts.traversals.items():
        harvest = counts.edge_harvest.get(edge, [])
        if (
            edge[0] in leaves
            and config.edge_traffic * counts.probes <= count
            and config.edge_excess <= len(harvest) - config.edge_rate * count
        ):
            return Ending(HARVEST_EDGE, list(harvest), edge)
    searches = counts.stretch_searches
    middle_rate = config.middle_rate(counts.stretch_search_queries / max(searches, 1))
    if config.search_share * counts.stretch_probes <= searches and _rate_above(
        config, middle_rate, searches, len(counts.stretch_middles)
    ):
        return Ending(HARVEST_MIDDLES, list(counts.stretch_middles), None)
    if succeeds_at(counts, config, counts.stretch_probes):
        return Ending(SUCCESS, [], None)
    return None


def succeeds_at(
    counts: SiteCounts, config: CounterexampleSearchConfig, stretch_probes: int
) -> bool:
    """The success test, were the stretch ``stretch_probes`` long."""
    side = rate_side(
        config.disagreement_rate,
        config.level,
        config.first_look,
        stretch_probes,
        counts.stretch_searches,
    )
    return side is False


def _rate_above(config: CounterexampleSearchConfig, theta, trials, hits) -> bool:
    return rate_side(theta, config.level, config.first_look, trials, hits) is True
