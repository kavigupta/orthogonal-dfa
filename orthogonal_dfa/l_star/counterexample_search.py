"""The counterexample search of ``OrthoDFA/TallyLoop.lean``, read through the round's suffix family."""

from collections import Counter
from dataclasses import dataclass
from itertools import islice
from typing import Dict, Iterable, List, Optional, Tuple

from automata.fa.dfa import DFA

from .counterexample_config import CounterexampleSearchConfig
from .counterexample_probe import (
    Charges,
    Cut,
    Edge,
    EdgeFound,
    Member,
    Outcome,
    Pair,
    Path,
    Triple,
    charges,
    locate,
)
from .midfix_tree import MidfixTree
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


def lcp(a: Path, b: Path) -> Path:
    out = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return tuple(out)


class CounterexampleSearch:
    """``TState``, over a tree it splits in place."""

    def __init__(self, config: CounterexampleSearchConfig, cut: Cut, tree: MidfixTree):
        self.config, self.cut, self.tree = config, cut, tree
        self.edges: Dict[Edge, Tuple[Path, bytes]] = {}
        self.version = 0
        self.records: Dict[Edge, List[Tuple[bytes, Path]]] = {}
        self.n = 0
        self.searches = 0
        self.middles: List[bytes] = []
        self.probes = 0
        self.start_harvest: List[bytes] = []
        self.charged: Counter = Counter()
        self.edge_harvest: Dict[Edge, List[bytes]] = {}
        self.paths = set(tree.leaf_paths())

    def probe(self, x: bytes) -> Tuple[Outcome, Charges]:
        located = locate(self.cut, self.tree, self.edges, self.config.k, x)
        return located.outcome, charges(located, self.config.k, x)

    def sift(self, z: bytes) -> Optional[Path]:
        return self.tree.route(self.cut, z)[1]

    def run(self, draws: Iterable[bytes]) -> Ending:
        """``step`` over the first ``max_probes`` of ``draws``."""
        for x in islice(draws, self.config.max_probes):
            ending = self.step(x)
            if ending is not None:
                return ending
        return Ending(OUT_OF_PROBES, [], None)

    def step(self, x: bytes) -> Optional[Ending]:
        outcome, charged = self.probe(x)
        self._charge(charged)
        self._act(outcome)
        return self._look() or self._settle_one()

    def records_at(self, edge: Edge, target: Path) -> int:
        return sum(1 for _, t in self.records.get(edge, ()) if t == target)

    def violation(self) -> Optional[Tuple[Edge, Path]]:
        """``Violates``: an edge with ``m`` records at a target it does not point at."""
        for edge, records in self.records.items():
            if edge[0] not in self.paths:
                continue
            current = self.edges.get(edge)
            for target, count in Counter(t for _, t in records).items():
                if count >= self.config.m and (current is None or current[0] != target):
                    return edge, target
        return None

    def _fresh(self):
        self.n = 0
        self.searches = 0
        self.middles = []

    def _set_edge(self, edge: Edge, value: Tuple[Path, bytes]):
        self._fresh()
        self.edges[edge] = value
        self.version += 1

    def _charge(self, charged: Charges):
        self.n += 1
        self.middles += charged.middle
        self.probes += 1
        self.start_harvest += charged.start
        self.charged.update(charged.traversals)
        for edge, strings in charged.undecided_by_edge.items():
            self.edge_harvest.setdefault(edge, []).extend(strings)

    def _act(self, outcome: Outcome):
        if isinstance(outcome, Member):
            self._set_edge(outcome.edge, (outcome.target, outcome.witness))
        elif isinstance(outcome, (EdgeFound, Pair, Triple)):
            self.searches += 1
            if isinstance(outcome, EdgeFound):
                edge = (outcome.source, outcome.letter)
                self.records.setdefault(edge, []).append(
                    (outcome.prefix, outcome.target)
                )

    def _above(self, theta, n, h) -> bool:
        return rate_side(theta, self.config.level, self.config.first_look, n, h) is True

    def settles_below(self, n) -> bool:
        c = self.config
        return (
            rate_side(c.disagreement_rate, c.level, c.first_look, n, self.searches)
            is False
        )

    def _look(self) -> Optional[Ending]:
        c = self.config
        if self._above(c.start_rate, self.probes, len(self.start_harvest)):
            return Ending(HARVEST_START, list(self.start_harvest), None)
        for edge, count in self.charged.items():
            harvest = self.edge_harvest.get(edge, [])
            if (
                edge[0] in self.paths
                and c.edge_traffic * self.probes <= count
                and c.edge_excess <= len(harvest) - c.edge_rate * count
            ):
                return Ending(HARVEST_EDGE, list(harvest), edge)
        if c.search_share * self.n <= self.searches and self._above(
            c.middle_rate, self.searches, len(self.middles)
        ):
            return Ending(HARVEST_MIDDLES, list(self.middles), None)
        if self.settles_below(self.n):
            return Ending(SUCCESS, [], None)
        return None

    def _settle_one(self) -> Optional[Ending]:
        found = self.violation()
        if found is None:
            return None
        self._fix(*found)
        if len(self.paths) > self.config.max_leaves:
            return Ending(TOO_BIG, [], None)
        return None

    def _fix(self, edge: Edge, target: Path):
        """``fixEdge``: split where the current target also has ``m`` records, else redirect."""
        witness = next((w for w, t in self.records[edge] if t == target), b"")
        current = self.edges.get(edge)
        if current is not None and self.records_at(edge, current[0]) >= self.config.m:
            source, letter = edge
            node = self.tree.midfix_at(lcp(target, current[0]))
            self._split(source, bytes([letter]) + node)
        else:
            self._set_edge(edge, (target, witness))

    def _split(self, leaf: Path, midfix: bytes):
        self.tree.split(self.tree.leaf_at(leaf), midfix)
        self.paths = set(self.tree.leaf_paths())
        edges = {}
        for (source, letter), (target, witness) in self.edges.items():
            if source[: len(leaf)] == leaf:
                continue
            if target == leaf:
                target = self.sift(witness + bytes([letter]))
                if target is None:
                    continue
            edges[(source, letter)] = (target, witness)
        self.edges = edges
        self.records = {}
        self._fresh()
        self.version += 1

    def to_dfa(self, alphabet_size: int, initial_state: int) -> DFA:
        """An unlearned edge loops on its leaf."""
        ids = {path: self.tree.leaf_at(path) for path in self.paths}
        transitions = {
            ids[path]: {
                c: ids[self.edges.get((path, c), (path,))[0]]
                for c in range(alphabet_size)
            }
            for path in self.paths
        }
        return DFA(
            states=set(ids.values()),
            input_symbols=set(range(alphabet_size)),
            transitions=transitions,
            initial_state=initial_state,
            final_states=self.tree.accepting_leaves(),
            allow_partial=False,
        )
