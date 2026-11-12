"""Sources of the prefixes a round's populations draw from."""

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np

from .counterexample_probe import Charges
from .counterexample_search import CounterexampleSearch
from .fallbacks.score_patience import SCORE_STALLED, ScorePatience
from .rejection_source import RejectionSource, proving_attempts
from .round_end import HARVEST_ENDS, HARVEST_STARTS, HARVEST_STUCK, Ending


class UniformSource:
    """The learner's own sampler.  Every draw is a prefix, so this never fails."""

    def __init__(self, pst):
        self._pst = pst

    def draw(self) -> bytes:
        return self._pst.sampler.sample(
            self._pst.rng, alphabet_size=self._pst.alphabet_size
        )

    @property
    def proven(self) -> bool:
        return True


class HarvestSource(RejectionSource):
    """More of a round's boundary population, found the way its strings were:
    each attempt reads a fresh draw by one of their provenances, chosen in
    proportion to how many of them it found."""

    def __init__(self, provenances, rng, *, known, acc_threshold, retirable):
        """``provenances`` maps each provenance to how many strings it found.
        Worth drawing on where an attempt turns up a boundary string at least
        ``1 - acc_threshold`` of the time, and not at half that: only a read that
        disagrees or cannot be placed turns one up, and a round the gate refuses
        can disagree on no more than that share of them.  A ``retirable`` source
        that runs dry has its population retired rather than ending the run."""
        super().__init__()
        self.retirable = retirable
        assert acc_threshold < 1, acc_threshold
        self._good = 1 - acc_threshold
        self._served.update(known)
        self._seen = set(known)
        self._provenances = list(provenances)
        counts = np.array([provenances[p] for p in self._provenances], dtype=float)
        self._weights = counts / counts.sum()
        self._rng = rng

    def attempt_draw(self) -> bool:
        provenance = self._provenances[
            self._rng.choice(len(self._provenances), p=self._weights)
        ]
        found = False
        strings, start = provenance.sample_from()
        for string in strings:
            if string in self._seen:
                # A repeat is not drawn again: its walk start is spelled out instead.
                provenance.round.spell_out_starts({start})
            else:
                self._seen.add(string)
                self._pool.append(string)
                found = True
        return found

    @property
    def proving(self) -> tuple:
        return proving_attempts(self._good, self._good / 2)

    @property
    def poor(self) -> float:
        return self._good / 2

    def source_repr(self) -> str:
        return "boundary"


def harvest_reader(
    ending: Ending, search: CounterexampleSearch
) -> Callable[[Charges], List[bytes]]:
    """What a probe of ``search`` reads undecided where ``ending`` harvested."""
    if ending.kind == HARVEST_ENDS:
        return lambda charged: charged.ends
    if ending.kind == HARVEST_STARTS:
        return lambda charged: charged.starts
    if ending.kind == HARVEST_STUCK:
        if search.member_fallback is not None:
            return search.member_fallback.stuck_harvest
        return lambda charged: charged.stuck
    if ending.kind == SCORE_STALLED:
        return ScorePatience.stalled_harvest
    return lambda charged: charged.middle


@dataclass(frozen=True, eq=False)
class Replay:
    """Fresh probes of a round's last hypothesis, kept as ``read`` reads them."""

    round: CounterexampleSearch
    read: Callable[[Charges], List[bytes]]
    draw: Callable[[], bytes]

    def sample(self) -> List[bytes]:
        return self.sample_from()[0]

    def sample_from(self) -> Tuple[List[bytes], bytes]:
        """``sample``, and the walk start of the probe it read."""
        x = self.draw()
        return self.read(self.round.probe(x)[1]), x[: self.round.walk_start(x)]
