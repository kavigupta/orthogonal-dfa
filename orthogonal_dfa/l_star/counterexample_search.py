"""``indecisiveLStarLevel5`` of ``OrthoDFA/Level5.lean``, read through the round's suffix family."""

import copy
from itertools import islice
from typing import Callable, Iterable, Optional, Sequence, Tuple

from .counterexample_config import CounterexampleSearchConfig
from .counterexample_probe import (
    Charges,
    EdgeFound,
    End,
    Located,
    Member,
    Outcome,
    charges,
    follow,
    locate,
)
from .hypothesis_dfa import Cut, HypothesisDFA, Path
from .midfix_tree import MidfixTree
from .round_end import (
    OUT_OF_PROBES,
    SCORE_STALLED,
    TOO_BIG,
    Ending,
    SiteCounts,
    ending_test,
)

#: Members of a leaf an undecided edge is read on before it is left undecided.
MEMBER_LIMIT = 20


class CounterexampleSearch:
    """``Level5State``, splitting at every contradiction (Kearns–Vazirani)."""

    def __init__(
        self,
        config: CounterexampleSearchConfig,
        cut: Cut,
        tree: MidfixTree,
        *,
        members: Sequence[bytes] = (),
        start: Optional[Callable[[HypothesisDFA], Optional[Path]]] = None,
        score: Optional[Callable[[HypothesisDFA], float]] = None,
        patience: Optional[int] = None,
    ):
        """``members`` are strings an edge can be read on in place of the probe's
        own prefix; ``start`` is ``locate``'s.  With ``score``, ``run`` leaves the
        hypothesis at the best-scoring one it saw, and with ``patience`` stops once
        that many probes pass without a better one."""
        self.config = config
        self.hypothesis = HypothesisDFA(cut, tree)
        self.counts = SiteCounts()
        self.probes = 0
        self.members = list(members)
        self.start = start
        self._witnesses = []
        self.score, self.patience = score, patience
        self.best_score, self._best, self._best_at = None, None, 0
        self._scored_version = None

    def run(self, draws: Iterable[bytes]) -> Ending:
        """``step`` over the first ``max_probes`` of ``draws``."""
        ending = Ending(OUT_OF_PROBES, [], None)
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
                ending = Ending(SCORE_STALLED, [], None)
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
        self._scored_version = self.hypothesis.version
        value = self.score(self.hypothesis)
        if self.best_score is None or value > self.best_score:
            self.best_score, self._best, self._best_at = (
                value,
                self.snapshot(),
                self.probes,
            )
            self._scored_version = self.hypothesis.version

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
        located = locate(self.hypothesis, self.config.k, x, self.start)
        if isinstance(located.outcome, End):
            located = self._read_on_members(located, x)
        return located.outcome, charges(located, self.config.k, x)

    def close(self, alphabet_size: int) -> None:
        """Learn every unlearned edge its source leaf's members can place: a split
        drops the split leaf's out-edges, and the round may end before probes
        relearn them."""
        hypothesis = self.hypothesis
        for source in sorted(hypothesis.leaves):
            for letter in range(alphabet_size):
                if (source, letter) in hypothesis.edges:
                    continue
                found = self._member_target(source, letter)
                if found is not None:
                    hypothesis.set_edge((source, letter), found)

    def _member_target(self, source: Path, letter: int):
        """``(target, member)`` for the first of up to ``MEMBER_LIMIT`` members of
        ``source`` the cut places under ``letter``, else ``None``."""
        hypothesis, tried = self.hypothesis, 0
        for member in self._witnesses[::-1] + self.members:
            if tried == MEMBER_LIMIT:
                break
            if hypothesis.sift(member) != source:
                continue
            tried += 1
            target = hypothesis.sift(member + bytes([letter]))
            if target is not None:
                return target, member
        return None

    def _read_on_members(self, located: Located, x: bytes) -> Located:
        """Learn the unlearned edge the probe's prefix read undecided on from a
        member of its source leaf instead, as ``EdgeResolver`` does.

        With ``k`` 0 every probe reaches the edges near the start on the same
        few short strings, so one of them read undecided would block its edge
        for good.
        """
        if not located.walk:
            return located
        k, hypothesis = self.config.k, self.hypothesis
        walk, step = follow(hypothesis.edges, located.walk[0], x[k:])
        if step is None:
            return located
        source, letter = walk[-1], x[k + step]
        found = self._member_target(source, letter)
        if found is None:
            return located
        target, member = found
        self._witnesses.append(member)
        return Located(
            Member((source, letter), target, member),
            located.walk,
            located.sifts,
            located.search_queries,
        )

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
