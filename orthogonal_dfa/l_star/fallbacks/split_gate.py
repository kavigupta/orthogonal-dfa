"""Reading a split's new node on other members of the leaf it splits before
splitting.  Not covered by the proof: Level5.lean splits on the probe's two
sifts alone."""

from itertools import islice
from typing import Iterator, Sequence

from ..counterexample_probe import EdgeFound, Located, Middle
from ..hypothesis_dfa import HypothesisDFA


class SplitGate:
    def __init__(self, members: Sequence[bytes], reads: int):
        self.members = list(members)
        self.reads = reads

    def read(self, hypothesis: HypothesisDFA, located: Located) -> Located:
        """An EdgeFound probe, left undecided at the edge's target where the new node
        reads one of up to self.reads other members of the source undecided.

        Where the source holds one state, those reads share the law of the read the
        split rests on, so a wrong read there gets past with chance

            (1 - U)^reads

        U being that law's chance of reading undecided.
        """
        found = located.outcome
        node = hypothesis.split_node(found.source, found.letter, found.target)
        for queries, member in enumerate(
            islice(self._members(hypothesis, found), self.reads), 1
        ):
            if hypothesis.cut(member + node) is None:
                return Located(
                    Middle(len(found.prefix) + 1, member + node),
                    located.walk,
                    located.sifts,
                    located.search_queries + queries,
                )
        return located

    def _members(self, hypothesis: HypothesisDFA, found: EdgeFound) -> Iterator[bytes]:
        """Strings the source sifts to, other than the probe's prefix and the edge's
        witness: the successors of the edges into it, which a split re-sifts anyway,
        then the witnesses out of it, then members."""
        source, edges = found.source, sorted(hypothesis.edges.items())
        skip = {found.prefix, hypothesis.edges[(source, found.letter)][1]}
        into = [w + bytes([c]) for (_, c), (t, w) in edges if t == source]
        out = [w for (s, _), (_, w) in edges if s == source]
        for member in dict.fromkeys(into + out + self.members):
            if member not in skip and hypothesis.sift(member) == source:
                yield member
