"""``indecisiveLStarLevel5`` of ``OrthoDFA/Level5.lean``, read through the round's suffix family."""

import copy
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
)
from .fallbacks.members import MemberFallback
from .hypothesis_dfa import Cut, HypothesisDFA, Path
from .midfix_tree import MidfixTree
from .round_end import (
    HARVESTS,
    OUT_OF_PROBES,
    SCORE_STALLED,
    SUCCESS,
    Ending,
    SiteCounts,
    ending_test,
    succeeds_at,
)


class CounterexampleSearch:
    """``Level5State``, splitting at every contradiction (Kearns–Vazirani)."""

    def __init__(
        self,
        config: CounterexampleSearchConfig,
        cut: Cut,
        tree: MidfixTree,
        *,
        start: Optional[Callable[[HypothesisDFA], Optional[Path]]] = None,
        score: Optional[Callable[[HypothesisDFA], float]] = None,
        patience: Optional[int] = None,
        max_k: int,
        member_fallback: Optional[MemberFallback] = None,
        harvest_tests: bool = True,
    ):
        """``start`` is ``locate``'s.  With ``score``, ``run`` leaves the
        hypothesis at the best-scoring one it saw, and with ``patience`` stops once
        that many probes pass without a better one.  A harvest that repeats a
        string increments ``k`` instead of ending the search, up to ``max_k``.
        Without ``harvest_tests`` only the success test ends it early."""
        self.config = config
        self.hypothesis = HypothesisDFA(cut, tree)
        self.counts = SiteCounts()
        self.probes = 0
        self.start = start
        self.score, self.patience = score, patience
        self.best_score, self._best, self._best_at = None, None, 0
        self._scored_version = None
        self.k, self.max_k = config.k, max_k
        #: Every distinct string read undecided, which a stalled score harvests.
        self.undecided = {}
        self.member_fallback = member_fallback
        self.harvest_tests = harvest_tests

    def run(self, draws: Iterable[bytes]) -> Ending:
        """``step`` over the first ``max_probes`` of ``draws``."""
        ending = Ending(OUT_OF_PROBES, [])
        self._track_best()
        for x in islice(draws, self.config.max_probes):
            found = self.step(x)
            self._track_best()
            if found is not None:
                ending = found
                break
            if (
                self.patience is not None
                and self.probes - self._best_at >= self.patience
            ):
                ending = Ending(SCORE_STALLED, list(self.undecided))
                break
        self.restore_best()
        return ending

    def snapshot(self):
        h = self.hypothesis
        return copy.deepcopy(h.tree), dict(h.edges), set(h.leaves)

    def restore(self, snap) -> None:
        h = self.hypothesis
        tree, edges, leaves = snap
        h.tree, h.edges, h.leaves = copy.deepcopy(tree), dict(edges), set(leaves)
        h.version += 1

    def restore_best(self) -> None:
        if self._best is not None:
            self.restore(self._best)

    def _track_best(self) -> None:
        if self.score is None or self.hypothesis.version == self._scored_version:
            return
        value = self.score(self.hypothesis)
        if self.best_score is None or value > self.best_score:
            self.best_score, self._best = value, self.snapshot()
            self._best_at = self.probes
        self._scored_version = self.hypothesis.version

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
        self.undecided.update(
            dict.fromkeys(charged.ends + charged.stuck + charged.middle)
        )
        if not self.harvest_tests:
            if succeeds_at(self.counts, self.config, self.counts.probes):
                return Ending(SUCCESS, [])
            return None
        ending = ending_test(self.counts, self.config, self.hypothesis.tree.depth)
        if ending is not None and self._deepens(ending):
            return None
        return ending

    def _deepens(self, ending: Ending) -> bool:
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
        located = locate(self.hypothesis, self.k, x, self.start)
        stuck = isinstance(located.outcome, End) and located.outcome.stuck
        if stuck and self.member_fallback is not None:
            located = self.member_fallback.read_stuck(
                self.hypothesis, self.k, located, x
            )
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
