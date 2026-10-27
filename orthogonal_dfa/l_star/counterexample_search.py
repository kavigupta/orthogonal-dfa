"""``indecisiveLStarLevel5`` of ``OrthoDFA/Level5.lean``, read through the round's suffix family."""

from itertools import islice
from typing import Iterable, Optional, Sequence, Tuple

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
from .round_end import HARVESTS, OUT_OF_PROBES, Ending, SiteCounts, ending_test

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
        max_k: int,
        members: Sequence[bytes] = (),
    ):
        """A harvest that repeats a string increments ``k`` instead of ending the
        search, up to ``max_k``.  ``members`` are strings an edge can be read on in
        place of the probe's own prefix."""
        self.config = config
        self.hypothesis = HypothesisDFA(cut, tree)
        self.counts = SiteCounts()
        self.probes = 0
        self.k, self.max_k = config.k, max_k
        self.members = list(members)
        self._witnesses = []

    def run(self, draws: Iterable[bytes]) -> Ending:
        """``step`` over the first ``max_probes`` of ``draws``."""
        for x in islice(draws, self.config.max_probes):
            ending = self.step(x)
            if ending is not None:
                return ending
        return Ending(OUT_OF_PROBES, [])

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
        located = locate(self.hypothesis, self.k, x)
        if isinstance(located.outcome, End) and located.outcome.stuck:
            located = self._read_on_members(located, x)
        return located.outcome, charges(located, self.k, x)

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
        """Learn the edge a probe is stuck at from a member of its source leaf,
        as ``EdgeResolver`` does.

        Every probe reaches the edges near the start on the same few short
        strings, so one of them read undecided would block its edge for good.
        """
        k = self.k
        walk, step = follow(self.hypothesis.edges, located.walk[0], x[k:])
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
