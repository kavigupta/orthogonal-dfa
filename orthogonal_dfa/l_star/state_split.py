"""Whether a hypothesis state holds members of the opposite label.

The empty suffix reads each member's own label, so across a state's members its
reads are independent of every other suffix's -- unless the state holds a minority
of the other label, which the class-preserving suffixes read at the other label's
rate too.  On half the members, the candidate suffixes that go with the empty suffix
best are picked out; on the other half, only those are read, and the members'
empty-suffix reads are asked whether they follow them.

The candidates are the pool and fresh draws.  The pool is screened towards the
class-preserving suffixes but nothing guarantees it holds them; the fresh draws do,
at the share the preconditions promise.

The minority worth finding is one of at least ``merged_minority_mass`` of the
sampler's mass.  How many members that takes depends on how well the picked
suffixes read a member's label, which only the data says, so the check starts at
the size a perfect reading would need and doubles while the answer is unclear.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import scipy.stats

from .dfa_utils import count_paths_to_state, uniform_weights
from .preconditions import DEFAULT_MIN_CLASS_PRESERVING_FRAC
from .prefix_sources import aim_at
from .rejection_source import _MISREAD, RejectionSource

#: The largest variance a read can have, at a rate of a half.
_MAX_READ_VARIANCE = 0.25


def tail_ladder(members, minority_share) -> List[int]:
    """Tail sizes to test, doubling from the least minority worth finding: a larger
    minority fills a tail of its own size, and a smaller tail holds an arbitrary
    part of it."""
    sizes = []
    size = max(1, math.ceil(minority_share * members))
    while size <= members // 2:
        sizes.append(size)
        size *= 2
    return sizes


def fresh_suffixes(miss_rate, separating=DEFAULT_MIN_CLASS_PRESERVING_FRAC) -> int:
    """Fewest fresh draws that hold two separating suffixes but with ``miss_rate``.

    Every class-preserving suffix separates two states of opposite labels, and the
    preconditions hold those to ``separating`` of the draws.
    """
    m = 2
    while scipy.stats.binom.cdf(1, m, separating) > miss_rate:
        m += 1
    return m


def _label_signal(signal, minority_share) -> float:
    """Squared correlation between a member's true label and its empty-suffix read,
    at the noisiest placement of the two rates."""
    spread = minority_share * (1 - minority_share)
    return spread * (2 * signal) ** 2 / _MAX_READ_VARIANCE


def first_look(signal, minority_share, level, miss_rate) -> int:
    """Members each half needs if the picked suffixes read every member's label
    perfectly, which no score can beat.  Sizing only: the test itself is exact."""
    z = scipy.stats.norm.isf(level) + scipy.stats.norm.isf(miss_rate)
    return math.ceil(z**2 / _label_signal(signal, minority_share))


def most_looks(signal, minority_share, separating, candidates) -> int:
    """Doublings from ``first_look`` to the size a score counting every candidate,
    unpicked, needs when ``separating`` of them separate."""
    # Squared correlation of that count with a member's label.
    carried = separating**2 * (2 * signal) ** 2 * minority_share * (1 - minority_share)
    quality = carried / (candidates * _MAX_READ_VARIANCE + carried)
    return math.ceil(math.log2(1 / quality)) + 1


def _tail(scores, size, rng, top) -> np.ndarray:
    """The ``size`` members ranked furthest to one end, ties broken at random."""
    order = np.lexsort((rng.random(len(scores)), -scores if top else scores))
    tail = np.zeros(len(scores), dtype=bool)
    tail[order[:size]] = True
    return tail


def _label_pvalue(tail, empty, top) -> float:
    """Exact one-sided p-value for the tail's empty-suffix reads leaning the way its
    score does.  The tail is a function of the score alone, and the score of other
    suffixes' reads, so given the counts the tail's ones are hypergeometric."""
    dist = scipy.stats.hypergeom(len(tail), int(empty.sum()), int(tail.sum()))
    ones = int(empty[tail].sum())
    return float(dist.sf(ones - 1) if top else dist.cdf(ones))


@dataclass
class StateSplit:
    """Two groups of a state's members, and what places another member in one."""

    #: ``groups[side]`` for ``side`` in (False, True); True is the minority.
    groups: Tuple[List[bytes], List[bytes]]
    suffixes: List[bytes]
    #: A member's score is its reads weighed by these, and the minority's lie at
    #: or above ``cut``.
    weights: np.ndarray
    cut: float

    def sides(self, prefixes, oracle) -> np.ndarray:
        return _reads(oracle, prefixes, self.suffixes) @ self.weights >= self.cut


def _reads(oracle, members, suffixes) -> np.ndarray:
    return np.asarray(
        oracle.membership_queries([p + v for p in members for v in suffixes]),
        dtype=np.int8,
    ).reshape(len(members), len(suffixes))


def _going_with(reads, empty) -> np.ndarray:
    """The suffixes to read on the held-out members, the ones going with the empty
    suffix best first.

    Ranked by the exact p-value of their ones falling with the empty suffix's, and
    cut where the count over those so far goes with the empty suffix most strongly.
    """
    dist = scipy.stats.hypergeom(len(empty), int(empty.sum()), reads.sum(0))
    order = np.argsort(dist.sf((reads & empty[:, None]).sum(0) - 1), kind="stable")
    counts = np.cumsum(reads[:, order], axis=1).astype(float)
    centred = counts - counts.mean(0)
    label = empty - empty.mean()
    spread = np.sqrt((centred**2).sum(0) * (label**2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        strength = np.where(spread > 0, centred.T @ label / spread, -np.inf)
    return order[: int(np.argmax(strength)) + 1]


def split_members(picking, testing, candidates, oracle, *, minority_share, level, rng):
    """One look: the split ``picking`` and ``testing`` show at ``level``, or
    ``None``, and the least p-value, corrected over the tails tried.

    Both tails of the score are tried, since the minority may be either label, at
    every size ``tail_ladder`` gives.  The split is the smallest tail that clears
    ``level``: a larger one holds the minority diluted.
    """
    suffixes = [v for v in candidates if v]
    members = picking + testing
    empty = np.asarray(oracle.membership_queries(members), dtype=np.int8)
    picks = np.arange(len(members)) < len(picking)
    picked_reads = _reads(oracle, picking, suffixes)
    picked = _going_with(picked_reads, empty[picks])
    chosen = [suffixes[k] for k in picked]
    held = _reads(oracle, testing, chosen).sum(1)
    tests = [
        (_label_pvalue(_tail(held, size, rng, top), empty[~picks], top), size, top)
        for size in tail_ladder(len(testing), minority_share)
        for top in (True, False)
    ]
    if not tests:
        return None, 1.0
    least = min(1.0, min(tests)[0] * len(tests))
    clearing = [(size, p, top) for p, size, top in tests if p * len(tests) <= level]
    if not clearing:
        return None, least
    size, _, top = min(clearing)
    weights = np.ones(len(chosen)) if top else -np.ones(len(chosen))
    scores = np.empty(len(members))
    scores[picks] = picked_reads[:, picked] @ weights
    scores[~picks] = held if top else -held
    # The minority is the same share of every member.
    inside = _tail(scores, math.ceil(size / len(testing) * len(members)), rng, True)
    split = StateSplit(
        groups=(
            [m for m, h in zip(members, inside) if not h],
            [m for m, h in zip(members, inside) if h],
        ),
        suffixes=chosen,
        weights=weights,
        cut=scores[inside].min(),
    )
    return split, least


def split_by_looks(draw, pool, oracle, *, signal, minority_share, alpha, rng):
    """The split of a state, or ``None``, from members ``draw(count)`` returns.

    Each look draws its own members, so the looks are independent and share
    ``alpha`` between them.  A look whose evidence is weaker than even odds calls
    the state one population; one in between doubles the members, since the picked
    suffixes read labels less well than ``first_look`` assumed.
    """
    fresh_count = fresh_suffixes(alpha)
    fresh = {draw.suffix() for _ in range(fresh_count)}
    candidates = sorted(set(pool) | fresh)
    looks = most_looks(signal, minority_share, 2, len(candidates))
    level = alpha / looks
    size = first_look(signal, minority_share, level, level)
    for _ in range(looks):
        split, p = split_members(
            draw(size),
            draw(size),
            candidates,
            oracle,
            minority_share=minority_share,
            level=level,
            rng=rng,
        )
        if split is not None:
            return split
        if p > 1 / 2:
            return None
        size *= 2
    return None


class _Aimed:
    """Draws for a hypothesis state: members aimed at it, and fresh suffixes."""

    def __init__(self, pst, aim):
        self._pst = pst
        self._aim = aim

    def __call__(self, count) -> List[bytes]:
        return [p for p in (self._aim() for _ in range(count)) if p is not None]

    def suffix(self) -> bytes:
        return self._pst.sampler.sample(self._pst.rng, alphabet_size=self._pst.alphabet_size)


def state_split(pst, dfa, state, *, alpha):
    """``split_by_looks`` over prefixes aimed at ``state``, and the aim that drew
    them."""
    aim = aim_at(pst, dfa, state)
    if aim is None:
        return None
    length = pst.sampler.length
    reaching = count_paths_to_state(dfa, state, length, uniform_weights(dfa))
    share = reaching[length][dfa.initial_state] / pst.alphabet_size**length
    minority_share = pst.config.merged_minority_mass / share
    if minority_share >= 1:
        # The whole state is lighter than the least minority worth finding.
        return None
    found = split_by_looks(
        _Aimed(pst, aim),
        [pst.table.suffix(v) for v in pst.table.fully_observed()],
        pst.table.memo,
        signal=pst.config.min_signal_strength,
        minority_share=minority_share,
        alpha=alpha,
        rng=pst.rng,
    )
    return None if found is None else (found, aim)


class SplitSource(RejectionSource):
    """More of one side of a split: aimed at the split state, kept where the split's
    score places them on this side.

    Proven on the members the split was found over, which were drawn by the same aim
    and placed the same way, so their count on this side is the yield test.  Called
    dry below the rate that count puts a floor under.
    """

    def __init__(self, split: StateSplit, side: bool, aim, oracle):
        super().__init__()
        self._split = split
        self._side = side
        self._aim = aim
        self._oracle = oracle
        on_side = len(split.groups[side])
        drawn = on_side + len(split.groups[not side])
        self._poor = float(scipy.stats.beta.ppf(_MISREAD, on_side, drawn - on_side + 1))
        self._pool.extend(split.groups[side])
        self._proven = True

    @property
    def proving(self) -> tuple:
        raise AssertionError("proven on construction; the yield test is never asked")

    @property
    def poor(self) -> float:
        return self._poor

    def attempt_draw(self) -> bool:
        prefix = self._aim()
        if self._split.sides([prefix], self._oracle)[0] != self._side:
            return False
        self._pool.append(prefix)
        return True

    def source_repr(self) -> str:
        return f"SplitSource(side={self._side}, over {len(self._split.suffixes)} suffixes)"
