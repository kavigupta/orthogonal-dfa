"""``indecisiveLStarLevel5`` of ``OrthoDFA/Level5.lean``, read through the round's suffix family."""

from itertools import islice
from typing import Iterable, Optional, Tuple

from .counterexample_config import CounterexampleSearchConfig
from .counterexample_probe import Charges, EdgeFound, Member, Outcome, charges, locate
from .hypothesis_dfa import Cut, HypothesisDFA, Path
from .midfix_tree import MidfixTree
from .round_end import OUT_OF_PROBES, TOO_BIG, Ending, SiteCounts, ending_test


class CounterexampleSearch:
    """``Level5State``, splitting at every contradiction (Kearns–Vazirani)."""

    def __init__(self, config: CounterexampleSearchConfig, cut: Cut, tree: MidfixTree):
        self.config = config
        self.hypothesis = HypothesisDFA(cut, tree)
        self.counts = SiteCounts()
        self.probes = 0

    def run(self, draws: Iterable[bytes]) -> Ending:
        """``step`` over the first ``max_probes`` of ``draws``."""
        for x in islice(draws, self.config.max_probes):
            ending = self.step(x)
            if ending is not None:
                return ending
        return Ending(OUT_OF_PROBES, [], None)

    def step(self, x: bytes) -> Optional[Ending]:
        outcome, charged = self.probe(x)
        self.probes += 1
        self.counts.add(charged, outcome)
        if isinstance(outcome, Member):
            self.counts = SiteCounts()
            self.hypothesis.set_edge(outcome.edge, (outcome.target, outcome.witness))
        hypothesis = self.hypothesis
        ending = ending_test(
            self.counts, self.config, hypothesis.tree.depth, hypothesis.leaves
        )
        if ending is None and isinstance(outcome, EdgeFound):
            return self._split(outcome)
        return ending

    def probe(self, x: bytes) -> Tuple[Outcome, Charges]:
        located = locate(self.hypothesis, self.config.k, x)
        return located.outcome, charges(located, self.config.k, x)

    def _split(self, found: EdgeFound) -> Optional[Ending]:
        """Split the edge's source on its letter and the midfix where the probe's
        target and the edge's part."""
        hypothesis = self.hypothesis
        current = hypothesis.edges[(found.source, found.letter)][0]
        midfix = hypothesis.tree.midfix_at(_common_prefix(found.target, current))
        hypothesis.split_leaf(found.source, bytes([found.letter]) + midfix)
        self.counts = SiteCounts()
        if len(hypothesis.leaves) > self.config.max_leaves:
            return Ending(TOO_BIG, [], None)
        return None


def _common_prefix(a: Path, b: Path) -> Path:
    out = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return tuple(out)
