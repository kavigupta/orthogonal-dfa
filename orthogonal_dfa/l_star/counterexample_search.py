"""``indecisiveLStarLevel5`` of ``OrthoDFA/Level5.lean``, read through the round's suffix family."""

from collections import Counter
from itertools import islice
from typing import Dict, Iterable, List, Optional, Tuple

from .counterexample_config import CounterexampleSearchConfig
from .counterexample_probe import Charges, EdgeFound, Member, Outcome, charges, locate
from .hypothesis_dfa import Cut, Edge, HypothesisDFA, Path
from .midfix_tree import MidfixTree
from .round_end import OUT_OF_PROBES, TOO_BIG, Ending, SiteCounts, ending_test


class CounterexampleSearch:
    """``Level5State``."""

    def __init__(self, config: CounterexampleSearchConfig, cut: Cut, tree: MidfixTree):
        self.config = config
        self.hypothesis = HypothesisDFA(cut, tree)
        self.counts = SiteCounts()
        self.records: Dict[Edge, List[Tuple[bytes, Path]]] = {}

    def run(self, draws: Iterable[bytes]) -> Ending:
        """``step`` over the first ``max_probes`` of ``draws``."""
        for x in islice(draws, self.config.max_probes):
            ending = self.step(x)
            if ending is not None:
                return ending
        return Ending(OUT_OF_PROBES, [], None)

    def step(self, x: bytes) -> Optional[Ending]:
        outcome, charged = self.probe(x)
        self.counts.add(charged, outcome)
        self._apply_outcome(outcome)
        hypothesis = self.hypothesis
        return (
            ending_test(
                self.counts, self.config, hypothesis.tree.depth, hypothesis.leaves
            )
            or self._fix_one_edge()
        )

    def probe(self, x: bytes) -> Tuple[Outcome, Charges]:
        located = locate(self.hypothesis, self.config.k, x)
        return located.outcome, charges(located, self.config.k, x)

    def _apply_outcome(self, outcome: Outcome):
        if isinstance(outcome, Member):
            self._set_edge(outcome.edge, (outcome.target, outcome.witness))
        elif isinstance(outcome, EdgeFound):
            edge = (outcome.source, outcome.letter)
            self.records.setdefault(edge, []).append((outcome.prefix, outcome.target))

    def _set_edge(self, edge: Edge, value: Tuple[Path, bytes]):
        self.counts.reset_stretch()
        self.hypothesis.set_edge(edge, value)

    def _fix_one_edge(self) -> Optional[Ending]:
        found = self.violation()
        if found is None:
            return None
        self._fix_edge(*found)
        if len(self.hypothesis.leaves) > self.config.max_leaves:
            return Ending(TOO_BIG, [], None)
        return None

    def violation(self) -> Optional[Tuple[Edge, Path]]:
        """``Violates``: an edge with ``m`` records at a target it does not point at."""
        for edge, records in self.records.items():
            if edge[0] not in self.hypothesis.leaves:
                continue
            current = self.hypothesis.edges.get(edge)
            for target, count in Counter(t for _, t in records).items():
                if count >= self.config.m and (current is None or current[0] != target):
                    return edge, target
        return None

    def _fix_edge(self, edge: Edge, target: Path):
        """``fixEdge``: split where the current target also has ``m`` records, else redirect."""
        witness = next((w for w, t in self.records[edge] if t == target), b"")
        current = self.hypothesis.edges.get(edge)
        if current is not None and self.records_at(edge, current[0]) >= self.config.m:
            source, letter = edge
            node = self.hypothesis.tree.midfix_at(_common_prefix(target, current[0]))
            self.hypothesis.split_leaf(source, bytes([letter]) + node)
            self.records = {}
            self.counts.reset_stretch()
        else:
            self._set_edge(edge, (target, witness))

    def records_at(self, edge: Edge, target: Path) -> int:
        return sum(1 for _, t in self.records.get(edge, ()) if t == target)


def _common_prefix(a: Path, b: Path) -> Path:
    out = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return tuple(out)
