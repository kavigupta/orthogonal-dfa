"""``indecisiveLStarLevel5`` of ``OrthoDFA/Level5.lean``, read through the round's suffix family."""

from itertools import islice
from typing import Iterable, Optional, Tuple

from .counterexample_config import CounterexampleSearchConfig
from .counterexample_probe import Charges, EdgeFound, Member, Outcome, charges, locate
from .hypothesis_dfa import Cut, HypothesisDFA, Path
from .midfix_tree import MidfixTree
from .round_end import HARVESTS, OUT_OF_PROBES, Ending, SiteCounts, ending_test


class CounterexampleSearch:
    """``Level5State``, splitting at every contradiction (Kearns–Vazirani)."""

    def __init__(
        self,
        config: CounterexampleSearchConfig,
        cut: Cut,
        tree: MidfixTree,
        *,
        max_k: int,
    ):
        """A harvest that repeats a string doubles ``k`` instead of ending the
        search, up to ``max_k``."""
        self.config = config
        self.hypothesis = HypothesisDFA(cut, tree)
        self.counts = SiteCounts()
        self.probes = 0
        self.k, self.max_k = config.k, max_k
        #: ``(probe, k)`` at each deepening.
        self.deepenings = []

    def run(self, draws: Iterable[bytes]) -> Ending:
        """``step`` over the first ``max_probes`` of ``draws``."""
        for x in islice(draws, self.config.max_probes):
            ending = self.step(x)
            if ending is not None:
                return ending
        return Ending(OUT_OF_PROBES, [])

    def step(self, x: bytes) -> Optional[Ending]:
        outcome, charged = self.probe(x)
        self.probes += 1
        if isinstance(outcome, Member):
            self.hypothesis.set_edge(outcome.edge, (outcome.target, outcome.witness))
            self.counts = SiteCounts()
            return None
        if isinstance(outcome, EdgeFound):
            self._split(outcome)
            self.counts = SiteCounts()
            return None
        self.counts.add(charged, outcome)
        ending = ending_test(self.counts, self.config, self.hypothesis.tree.depth)
        if ending is not None and self._deepens(ending):
            return None
        return ending

    def _deepens(self, ending: Ending) -> bool:
        """Walk deeper in place of a harvest that repeats a string, which a deeper
        start stops reaching.  ``Level5.lean`` holds ``k`` fixed for the round."""
        if (
            self.k >= self.max_k
            or ending.kind not in HARVESTS
            or len(set(ending.harvest)) == len(ending.harvest)
        ):
            return False
        self.k = min(max(2 * self.k, 1), self.max_k)
        self.counts = SiteCounts()
        self.deepenings.append((self.probes, self.k))
        print(f"  {ending.kind} repeated a string; k now {self.k}")
        return True

    def probe(self, x: bytes) -> Tuple[Outcome, Charges]:
        located = locate(self.hypothesis, self.k, x)
        return located.outcome, charges(located, self.k, x)

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
