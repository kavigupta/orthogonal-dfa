"""indecisiveLStarLevel3 of RLStar/Code/Level3.lean, read through the round's suffix family."""

from itertools import islice
from typing import Callable, Iterable, Optional, Tuple

from .counterexample_config import CounterexampleSearchConfig
from .counterexample_probe import (
    Charges,
    EdgeFound,
    End,
    Member,
    Outcome,
    charges,
    locate,
    plain_bisect,
)
from .fallbacks.members import MemberFallback
from .fallbacks.score_patience import ScorePatience
from .hypothesis_dfa import Cut, HypothesisDFA, Path
from .midfix_tree import MidfixTree
from .round_end import OUT_OF_PROBES, Ending, SiteCounts, ending_test
from .trie import Trie


class CounterexampleSearch:
    """``Level5State``, splitting at every contradiction (Kearns–Vazirani)."""

    def __init__(
        self,
        config: CounterexampleSearchConfig,
        cut: Cut,
        tree: MidfixTree,
        *,
        max_k: int,
        member_fallback: Optional[MemberFallback] = None,
        bisect: Callable = plain_bisect,
        score_patience: Optional[ScorePatience] = None,
    ):
        self.config = config
        self.hypothesis = HypothesisDFA(cut, tree)
        self.trie = Trie()
        self.counts = SiteCounts()
        self.probes = 0
        self.max_k = max_k
        self.member_fallback = member_fallback
        self.bisect = bisect
        self.score_patience = score_patience

    def run(self, draws: Iterable[bytes]) -> Ending:
        """``step`` over the first ``max_probes`` of ``draws``."""
        for x in islice(draws, self.config.max_probes):
            ending = self.step(x)
            if ending is not None:
                return ending
        return Ending(OUT_OF_PROBES, [])

    def step(self, x: bytes) -> Optional[Ending]:
        if self.score_patience is not None:
            stalled = self.score_patience.observe(self)
            if stalled is not None:
                return stalled
        start = x[: self.walk_start(x)]
        outcome, charged = self.probe(x)
        self.probes += 1
        if self._skips_repeat(charged, start):
            return None
        if isinstance(outcome, Member):
            self.hypothesis.set_edge(outcome.edge, (outcome.target, outcome.witness))
            self.counts = SiteCounts()
            return None
        if isinstance(outcome, EdgeFound):
            self._split(outcome)
            self.counts = SiteCounts()
            return None
        self.counts.add(charged, outcome, start)
        ending = ending_test(self.counts, self.config, self.hypothesis.tree.depth)
        if self.score_patience is not None:
            self.score_patience.record(charged)
        if self.score_patience is not None:
            ending = self.score_patience.filter(ending, self.counts, self.config)
        return ending

    def walk_start(self, x: bytes) -> int:
        """Past every spelled-out prefix of x."""
        return self.trie.walk_start(x)

    def _skips_repeat(self, charged: Charges, start: bytes) -> bool:
        """A probe reading a harvest string already read is not counted, nor is the
        probe that first read it; both walk starts are spelled out."""
        repeated = [
            z
            for z in charged.ends + charged.middle
            if z in self.counts.first or z in self.counts.repeated
        ]
        if not repeated:
            return False
        starts = {start}
        for z in repeated:
            starts |= self.counts.origins.get(z, set())
            self.counts.drop(z)
        self.spell_out_starts(starts)
        return True

    def spell_out_starts(self, starts):
        starts = {u for u in starts if len(u) < self.max_k} - self.trie.prefixes
        if starts:
            self.trie.spell_out(starts)

    def probe(self, x: bytes) -> Tuple[Outcome, Charges]:
        k = self.walk_start(x)
        located = locate(self.hypothesis, k, x, bisect=self.bisect)
        stuck = isinstance(located.outcome, End) and located.outcome.stuck
        if stuck and self.member_fallback is not None:
            located = self.member_fallback.read_stuck(self.hypothesis, k, located, x)
        return located.outcome, charges(located, k, x)

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
