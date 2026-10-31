"""Reading an edge a probe leaves undecided on a member of its source leaf, as
``EdgeResolver`` does.  Not covered by the proof: ``Level5.lean`` learns an edge
only from the probe's own prefix."""

from typing import List, Sequence

from ..counterexample_probe import Charges, Located, Member, follow
from ..hypothesis_dfa import HypothesisDFA, Path

#: Members of a leaf an undecided edge is read on before it is left undecided.
MEMBER_LIMIT = 20


class MemberFallback:
    def __init__(self, members: Sequence[bytes]):
        self.members = list(members)
        self._witnesses = []

    def read_stuck(
        self, hypothesis: HypothesisDFA, k: int, located: Located, x: bytes
    ) -> Located:
        """``located``, stuck at an unlearned edge, as a ``Member`` learning that
        edge where a member of its source leaf places it.

        Every probe reaches the edges near the start on the same few short
        strings, so one of them read undecided would block its edge for good.
        """
        walk, step = follow(hypothesis.edges, located.walk[0], x[k:])
        source, letter = walk[-1], x[k + step]
        found = self._target(hypothesis, source, letter)
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

    def close(self, hypothesis: HypothesisDFA, alphabet_size: int) -> None:
        """Learn every unlearned edge its source leaf's members can place: a split
        drops the split leaf's out-edges, and the round may end before probes
        relearn them."""
        for source in sorted(hypothesis.leaves):
            for letter in range(alphabet_size):
                if (source, letter) in hypothesis.edges:
                    continue
                found = self._target(hypothesis, source, letter)
                if found is not None:
                    hypothesis.set_edge((source, letter), found)

    @staticmethod
    def stuck_harvest(charged: Charges) -> List[bytes]:
        # Stuck reads do not recur: a later probe's members, tried witnesses
        # first, place the edge.  Every stuck read is also an end.
        return charged.ends

    def _target(self, hypothesis: HypothesisDFA, source: Path, letter: int):
        """``(target, member)`` for the first of up to ``MEMBER_LIMIT`` members of
        ``source`` the cut places under ``letter``, else ``None``."""
        tried = 0
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
