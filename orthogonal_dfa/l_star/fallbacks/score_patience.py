"""Ending a round's search on a held-out score: it keeps the best hypothesis it
saw, and stops once the score has not improved for a number of probes.  Not
covered by the proof: ``Level5.lean`` has no score."""

import copy
from typing import TYPE_CHECKING, Callable, List, NamedTuple, Optional

from ..counterexample_config import CounterexampleSearchConfig
from ..counterexample_probe import Charges
from ..hypothesis_dfa import HypothesisDFA
from ..round_end import HARVESTS, SUCCESS, Ending, SiteCounts, succeeds_at

if TYPE_CHECKING:
    from ..counterexample_search import CounterexampleSearch

SCORE_STALLED = "score stalled"


class Selection(NamedTuple):
    """Held-out ``(fit, read)`` pairs: ``search`` is what a search stops and reverts
    on, ``pick`` what restarts and rounds are ranked on, so the ranking is not biased
    by the stopping."""

    search: tuple
    pick: tuple


def snapshot(hypothesis: HypothesisDFA):
    return (
        copy.deepcopy(hypothesis.tree),
        dict(hypothesis.edges),
        set(hypothesis.leaves),
    )


def restore(hypothesis: HypothesisDFA, snap) -> None:
    tree, edges, leaves = snap
    hypothesis.tree = copy.deepcopy(tree)
    hypothesis.edges, hypothesis.leaves = dict(edges), set(leaves)
    hypothesis.version += 1


class ScorePatience:
    def __init__(
        self,
        score: Callable[[HypothesisDFA], float],
        patience: int,
        *,
        harvest_tests: bool,
    ):
        """With ``harvest_tests`` off only the success test or the stalled score
        ends a search."""
        self.score, self.patience = score, patience
        self.harvest_tests = harvest_tests
        self.best_score, self._best, self._best_at = None, None, 0
        self._scored_version = None
        #: Every distinct string read undecided, which a stalled score harvests.
        self.undecided = {}

    @classmethod
    def from_config(
        cls, config, score: Callable[[HypothesisDFA], float]
    ) -> Optional["ScorePatience"]:
        """``None`` where ``config`` sets no patience."""
        patience = config.counterexample_score_patience
        harvest_tests = config.counterexample_harvest_tests
        assert harvest_tests or patience is not None, "nothing would end the search"
        if patience is None:
            return None
        return cls(score, patience, harvest_tests=harvest_tests)

    def record(self, charged: Charges) -> None:
        self.undecided.update(dict.fromkeys(self.stalled_harvest(charged)))

    def observe(self, search: "CounterexampleSearch") -> Optional[Ending]:
        """Keep a snapshot of a new best hypothesis; the stalled ending once
        ``patience`` probes pass without one."""
        hypothesis = search.hypothesis
        if hypothesis.version != self._scored_version:
            value = self.score(hypothesis)
            if self.best_score is None or value > self.best_score:
                self.best_score, self._best = value, snapshot(hypothesis)
                self._best_at = search.probes
            self._scored_version = hypothesis.version
        if search.probes - self._best_at >= self.patience:
            return Ending(SCORE_STALLED, list(self.undecided))
        return None

    def finish(self, search: "CounterexampleSearch") -> None:
        """Leave ``search`` at the best hypothesis it saw, its last one included."""
        self.observe(search)
        restore(search.hypothesis, self._best)

    def filter(
        self,
        ending: Optional[Ending],
        counts: SiteCounts,
        config: CounterexampleSearchConfig,
    ) -> Optional[Ending]:
        """``ending``, or without ``harvest_tests`` success where a harvest fired
        on a hypothesis that succeeds."""
        if self.harvest_tests or ending is None or ending.kind not in HARVESTS:
            return ending
        if succeeds_at(counts, config, counts.probes):
            return Ending(SUCCESS, [])
        return None

    def guarded(self, hypothesis: HypothesisDFA, change: Callable[[], None]) -> None:
        """``change`` the hypothesis, undone if it lowers the score."""
        before = snapshot(hypothesis)
        change()
        if self.score(hypothesis) < self.best_score:
            restore(hypothesis, before)

    @staticmethod
    def stalled_harvest(charged: Charges) -> List[bytes]:
        return charged.ends + charged.stuck + charged.middle
