"""``indecisiveLStarLevel5`` of ``OrthoDFA/Level5.lean``, read through the round's suffix family."""

from itertools import islice
from typing import Iterable, List, Optional, Tuple

from .counterexample_config import CounterexampleSearchConfig
from .counterexample_probe import Charges, EdgeFound, Member, Outcome, charges, locate
from .hypothesis_dfa import Cut, HypothesisDFA, Path
from .midfix_tree import MidfixTree
from .round_end import OUT_OF_PROBES, Ending, SiteCounts, ending_test


class CounterexampleSearch:
    """``Level5State``, splitting at every contradiction (Kearns–Vazirani)."""

    def __init__(self, config: CounterexampleSearchConfig, cut: Cut, tree: MidfixTree):
        self.config = config
        self.hypothesis = HypothesisDFA(cut, tree)
        self.counts = SiteCounts()
        self.probes = 0

    def run(self, draws: Iterable[bytes]) -> Ending:
        """step over the first max_probes of draws, counting afresh."""
        self.counts = SiteCounts()
        for x in islice(draws, self.config.max_probes):
            ending = self.step(x)
            if ending is not None:
                return ending
        return Ending(OUT_OF_PROBES, [])

    def step(self, x: bytes) -> Optional[Ending]:
        outcome, charged = self.probe(x)
        self.probes += 1
        if self._learn(outcome):
            return None
        self.counts.add(charged, outcome)
        return ending_test(self.counts, self.config, self.hypothesis.tree.depth)

    def settle(self, xs: List[bytes]):
        """Probe xs from ε until a pass over them changes nothing, using
        only their outcomes: they were chosen for failing, so no rate test may
        count them."""
        start, version = self.probes, None
        while version != self.hypothesis.version and (
            self.probes - start < self.config.max_probes
        ):
            version = self.hypothesis.version
            for x in xs:
                self.probes += 1
                self._learn(locate(self.hypothesis, 0, x).outcome)

    def probe(self, x: bytes) -> Tuple[Outcome, Charges]:
        located = locate(self.hypothesis, self.config.k, x)
        return located.outcome, charges(located, self.config.k, x)

    def _learn(self, outcome: Outcome) -> bool:
        """Whether outcome changed the hypothesis, which starts the counts
        afresh."""
        if isinstance(outcome, Member):
            self.hypothesis.set_edge(outcome.edge, (outcome.target, outcome.witness))
        elif isinstance(outcome, EdgeFound):
            self._split(outcome)
        else:
            return False
        self.counts = SiteCounts()
        return True

    def _split(self, found: EdgeFound):
        """Split the edge's source on its letter and the midfix where the probe's
        target and the edge's part."""
        hypothesis = self.hypothesis
        current = hypothesis.edges[(found.source, found.letter)][0]
        midfix = hypothesis.tree.midfix_at(_common_prefix(found.target, current))
        hypothesis.split_leaf(found.source, bytes([found.letter]) + midfix)


def _common_prefix(a: Path, b: Path) -> Path:
    out = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return tuple(out)
