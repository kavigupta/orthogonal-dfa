"""Whether a hypothesis state holds members of the opposite label.

For members p of the state, with E_p the read of p itself and X_pv the read of p v:
if the state is one class, E_p is independent of every X_pv across members.  A
minority of the other label makes E depend on the class-preserving v's.  The test
picks candidate suffixes v on one half of the members and rejects

    H0: E independent of sum over picked v of X_pv

on the other half, exactly, at a level spread over looks of doubling size.
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
    """[ceil(w n) 2^k for k >= 0, while at most n / 2], w = ``minority_share``,
    n = ``members``."""
    sizes = []
    size = max(1, math.ceil(minority_share * members))
    while size <= members // 2:
        sizes.append(size)
        size *= 2
    return sizes


def fresh_suffixes(miss_rate, separating=DEFAULT_MIN_CLASS_PRESERVING_FRAC) -> int:
    """The least m with

        P(Binomial(m, separating) <= 1) <= miss_rate,

    ``separating`` being the class-preserving share the preconditions promise."""
    m = 2
    while scipy.stats.binom.cdf(1, m, separating) > miss_rate:
        m += 1
    return m


def _label_signal(signal, minority_share) -> float:
    """w (1 - w) (2 signal)^2 / v_max, w = ``minority_share``: the squared
    correlation between a member's label and its own read, the labels reading at
    1/2 -+ signal."""
    spread = minority_share * (1 - minority_share)
    return spread * (2 * signal) ** 2 / _MAX_READ_VARIANCE


def first_look(signal, minority_share, level, miss_rate) -> int:
    """ceil((z_level + z_miss)^2 / ``_label_signal``), the members per half at
    which a test on each member's true label would reach power 1 - ``miss_rate``
    at ``level``.  Sizing only, by the normal approximation; the test is exact."""
    z = scipy.stats.norm.isf(level) + scipy.stats.norm.isf(miss_rate)
    return math.ceil(z**2 / _label_signal(signal, minority_share))


def most_looks(signal, minority_share, separating, candidates) -> int:
    """ceil(log2(1 / q)) + 1, with

        q = m^2 (2 signal)^2 w (1 - w) / (M v_max + m^2 (2 signal)^2 w (1 - w))

    the squared correlation with a member's label of its count of ones over all
    M = ``candidates``, of which m = ``separating`` separate."""
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
    """P(H >= h) if ``top`` else P(H <= h), H ~ Hypergeometric(n, sum(E), |T|),
    h = sum over the tail T of E: exact when T is independent of E."""
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
    """Candidates ordered by P(H_v >= co-ones of v with E), H_v hypergeometric
    on the picking members, cut to the leading K maximising

        corr(sum over the first K of X_.v, E)."""
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
    """(split, p*): p* = min over tail sizes t in ``tail_ladder`` and both ends
    of T * ``_label_pvalue``, T the number of such tails, on the ``testing``
    members' counts over the suffixes ``_going_with`` picks on ``picking``.  The
    split is at the least t with T p <= ``level``, or ``None``."""
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


def split_by_looks(draw, family, oracle, *, signal, minority_share, alpha, rng):
    """The split from looks k = 0, 1, ..., L - 1 on fresh members, n_0 2^k per
    half with n_0 = ``first_look`` and L = ``most_looks``, each at level
    ``alpha`` / L: the first look's split, or ``None`` once a look's p* > 1/2."""
    fresh_count = fresh_suffixes(alpha)
    fresh = {draw.suffix() for _ in range(fresh_count)}
    candidates = sorted(set(family) | fresh)
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
        return self._pst.sampler.sample(
            self._pst.rng, alphabet_size=self._pst.alphabet_size
        )


def state_split(pst, dfa, state, family, *, alpha):
    """``split_by_looks`` over prefixes aimed at ``state`` and the suffixes
    ``family``, and the aim that drew the prefixes."""
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
        [pst.table.suffix(v) for v in family],
        # Every look reads fresh members, which the memo would only accumulate.
        pst.oracle,
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
        return (
            f"SplitSource(side={self._side}, over {len(self._split.suffixes)} suffixes)"
        )
