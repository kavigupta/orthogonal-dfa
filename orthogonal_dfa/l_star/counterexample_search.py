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


class CounterexampleSearch:
    """``TState``, over a tree it splits in place."""

    def __init__(self, config: CounterexampleSearchConfig, cut: Cut, tree: MidfixTree):
        self.config, self.cut, self.tree = config, cut, tree
        self.leaves = set(tree.leaf_paths())
        self.edges: Dict[Edge, Tuple[Path, bytes]] = {}
        self.version = 0
        self.records: Dict[Edge, List[Tuple[bytes, Path]]] = {}
        self.probes = 0
        self.start_harvest: List[bytes] = []
        self.traversals: Counter = Counter()
        self.edge_harvest: Dict[Edge, List[bytes]] = {}
        #: Since the last change: probes, those that searched, and their undecided middles.
        self.stretch_probes = 0
        self.stretch_searches = 0
        self.stretch_middles: List[bytes] = []

    def run(self, draws: Iterable[bytes]) -> Ending:
        """``step`` over the first ``max_probes`` of ``draws``."""
        for x in islice(draws, self.config.max_probes):
            ending = self.step(x)
            if ending is not None:
                return ending
        return Ending(OUT_OF_PROBES, [], None)

    def step(self, x: bytes) -> Optional[Ending]:
        outcome, charged = self.probe(x)
        self._count_probe(charged)
        self._apply_outcome(outcome)
        return self._ending_test() or self._fix_one_edge()

    def probe(self, x: bytes) -> Tuple[Outcome, Charges]:
        located = locate(self.cut, self.tree, self.edges, self.config.k, x)
        return located.outcome, charges(located, self.config.k, x)

    def _count_probe(self, charged: Charges):
        self.probes += 1
        self.stretch_probes += 1
        self.start_harvest += charged.start
        self.stretch_middles += charged.middle
        self.traversals.update(charged.traversals)
        for edge, strings in charged.undecided_by_edge.items():
            self.edge_harvest.setdefault(edge, []).extend(strings)

    def _apply_outcome(self, outcome: Outcome):
        if isinstance(outcome, Member):
            self._set_edge(outcome.edge, (outcome.target, outcome.witness))
        elif isinstance(outcome, (EdgeFound, Pair, Triple)):
            self.stretch_searches += 1
            if isinstance(outcome, EdgeFound):
                edge = (outcome.source, outcome.letter)
                self.records.setdefault(edge, []).append(
                    (outcome.prefix, outcome.target)
                )

    def _set_edge(self, edge: Edge, value: Tuple[Path, bytes]):
        self._start_new_stretch()
        self.edges[edge] = value
        self.version += 1

    def _start_new_stretch(self):
        self.stretch_probes = 0
        self.stretch_searches = 0
        self.stretch_middles = []

    def _ending_test(self) -> Optional[Ending]:
        """The harvest tests, then the success test."""
        config = self.config
        if self._rate_above(config.start_rate, self.probes, len(self.start_harvest)):
            return Ending(HARVEST_START, list(self.start_harvest), None)
        for edge, count in self.traversals.items():
            harvest = self.edge_harvest.get(edge, [])
            if (
                edge[0] in self.leaves
                and config.edge_traffic * self.probes <= count
                and config.edge_excess <= len(harvest) - config.edge_rate * count
            ):
                return Ending(HARVEST_EDGE, list(harvest), edge)
        if config.search_share * self.stretch_probes <= self.stretch_searches and (
            self._rate_above(
                config.middle_rate, self.stretch_searches, len(self.stretch_middles)
            )
        ):
            return Ending(HARVEST_MIDDLES, list(self.stretch_middles), None)
        if self.succeeds_at(self.stretch_probes):
            return Ending(SUCCESS, [], None)
        return None

    def _rate_above(self, theta, trials, hits) -> bool:
        config = self.config
        return rate_side(theta, config.level, config.first_look, trials, hits) is True

    def succeeds_at(self, stretch_probes) -> bool:
        """The success test, were the stretch ``stretch_probes`` long."""
        config = self.config
        side = rate_side(
            config.disagreement_rate,
            config.level,
            config.first_look,
            stretch_probes,
            self.stretch_searches,
        )
        return side is False

    def _fix_one_edge(self) -> Optional[Ending]:
        found = self.violation()
        if found is None:
            return None
        self._fix_edge(*found)
        if len(self.leaves) > self.config.max_leaves:
            return Ending(TOO_BIG, [], None)
        return None

    def violation(self) -> Optional[Tuple[Edge, Path]]:
        """``Violates``: an edge with ``m`` records at a target it does not point at."""
        for edge, records in self.records.items():
            if edge[0] not in self.leaves:
                continue
            current = self.edges.get(edge)
            for target, count in Counter(t for _, t in records).items():
                if count >= self.config.m and (current is None or current[0] != target):
                    return edge, target
        return None

    def _fix_edge(self, edge: Edge, target: Path):
        """``fixEdge``: split where the current target also has ``m`` records, else redirect."""
        witness = next((w for w, t in self.records[edge] if t == target), b"")
        current = self.edges.get(edge)
        if current is not None and self.records_at(edge, current[0]) >= self.config.m:
            source, letter = edge
            node = self.tree.midfix_at(_common_prefix(target, current[0]))
            self._split_leaf(source, bytes([letter]) + node)
        else:
            self._set_edge(edge, (target, witness))

    def records_at(self, edge: Edge, target: Path) -> int:
        return sum(1 for _, t in self.records.get(edge, ()) if t == target)

    def _split_leaf(self, leaf: Path, midfix: bytes):
        self.tree.split(self.tree.leaf_at(leaf), midfix)
        self.leaves = set(self.tree.leaf_paths())
        self._retarget_edges(leaf)
        self.records = {}
        self._start_new_stretch()
        self.version += 1

    def _retarget_edges(self, leaf: Path):
        """``retargetBy``: drop the split leaf's out-edges, and re-sift the
        witnesses of the edges into it."""
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

    def sift(self, z: bytes) -> Optional[Path]:
        return self.tree.sift_with_path(self.cut, z)[1]

    def to_dfa(self, alphabet_size: int, initial_state: int) -> DFA:
        """An unlearned edge loops on its leaf."""
        ids = {path: self.tree.leaf_at(path) for path in self.leaves}
        transitions = {
            ids[path]: {
                c: ids[self.edges.get((path, c), (path,))[0]]
                for c in range(alphabet_size)
            }
            for path in self.leaves
        }
        return DFA(
            states=set(ids.values()),
            input_symbols=set(range(alphabet_size)),
            transitions=transitions,
            initial_state=initial_state,
            final_states=self.tree.accepting_leaves(),
            allow_partial=False,
        )


def _common_prefix(a: Path, b: Path) -> Path:
    out = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return tuple(out)
