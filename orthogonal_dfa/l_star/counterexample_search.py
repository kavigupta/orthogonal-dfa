"""indecisiveLStarLevel3 of RLStar/Code/Level3.lean, read through the round's suffix family."""

from collections import Counter
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
from .round_end import HARVESTS, OUT_OF_PROBES, Ending, SiteCounts, ending_test

#: A harvested string at least this share of its harvest is spelled out, not harvested.
HEAVY_SHARE = 1 / 4


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
        """A harvest that repeats a string deepens the walk start instead of ending
        the search, up to ``max_k``."""
        self.config = config
        self.hypothesis = HypothesisDFA(cut, tree)
        self.counts = SiteCounts()
        self.probes = 0
        self.k, self.max_k = config.k, max_k
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
        # Before the filter: without harvest tests a harvest still deepens.
        if ending is not None and self._deepens(ending):
            return None
        if self.score_patience is not None:
            ending = self.score_patience.filter(ending, self.counts, self.config)
        return ending

    def walk_start(self, x: bytes) -> int:
        """Past every expanded prefix of ``x``, with local starts."""
        if not self.config.local_starts:
            return self.k
        expanded, k = self.hypothesis.expanded, 0
        while k < len(x) and x[:k] in expanded:
            k += 1
        return k

    def _deepens(self, ending: Ending) -> bool:
        if self.config.local_starts:
            return self._expands(ending)
        return self._deepens_globally(ending)

    def _expands(self, ending: Ending) -> bool:
        """Expand the walk starts of the probes whose harvested string repeats and is
        at least ``HEAVY_SHARE`` of the harvest; the rest of it is harvested."""
        if ending.kind not in HARVESTS:
            return False
        heavy = [
            z
            for z, n in Counter(ending.harvest).items()
            if n > 1 and n >= HEAVY_SHARE * len(ending.harvest)
        ]
        starts = {
            u
            for z in heavy
            for u in self.counts.origins.get(z, ())
            if len(u) < self.max_k
        }
        if not starts:
            return False
        self.hypothesis.expand(starts)
        self.k = 1 + max(len(u) for u in self.hypothesis.expanded)
        self.counts = SiteCounts()
        print(
            f"  {ending.kind} has a heavy string; expanded {sorted(starts)}, "
            f"{len(self.hypothesis.expanded)} expanded"
        )
        return True

    def _deepens_globally(self, ending: Ending) -> bool:
        """Walk deeper in place of a harvest that repeats a string, which a deeper
        start stops reaching.  The counts restart, or the test that just fired would
        fire again at once."""
        if (
            self.k >= self.max_k
            or ending.kind not in HARVESTS
            or len(set(ending.harvest)) == len(ending.harvest)
        ):
            return False
        self.k = min(self.k + 1, self.max_k)
        self.counts = SiteCounts()
        print(f"  {ending.kind} repeated a string; k now {self.k}")
        return True

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
