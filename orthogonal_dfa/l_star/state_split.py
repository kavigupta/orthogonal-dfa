"""Splitting a hypothesis state the certificate blames into its two labels.

For members p of the state, with E_p the read of p itself and X_pv the read of
p v, suffixes v whose reads go with E across the members separate a minority of
the other label from the rest, since a member's own read and a class-preserving
suffix's both follow its label.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import scipy.stats

from .prefix_sources import aim_at
from .rejection_source import _MISREAD, RejectionSource

#: The largest variance a read can have, at a rate of a half.
_MAX_READ_VARIANCE = 0.25


#: Separating suffixes the fresh draws are sized to hold.
_SEPARATING = 2


def fresh_suffixes(miss_rate, preserving_share) -> int:
    """The least m with

        P(Binomial(m, preserving_share) < _SEPARATING) <= miss_rate,

    preserving_share being the least class-preserving share of the sampler's
    draws."""
    m = _SEPARATING
    while scipy.stats.binom.cdf(_SEPARATING - 1, m, preserving_share) > miss_rate:
        m += 1
    return m


def _label_signal(signal, minority_share) -> float:
    """w (1 - w) (2 signal)^2 / v_max, w = minority_share: a lower bound on the
    squared correlation between a member's label and its own read, the labels
    reading at rates 2 signal apart."""
    spread = minority_share * (1 - minority_share)
    return spread * (2 * signal) ** 2 / _MAX_READ_VARIANCE


def first_look(signal, minority_share, level, miss_rate) -> int:
    """ceil((z_level + z_miss)^2 / _label_signal), the members per half at
    which a test on each member's true label would reach power 1 - miss_rate
    at level.  Sizing only, by the normal approximation; the test is exact."""
    z = scipy.stats.norm.isf(level) + scipy.stats.norm.isf(miss_rate)
    return math.ceil(z**2 / _label_signal(signal, minority_share))


def _tail(scores, size, rng, top) -> np.ndarray:
    """A mask of the members ranked furthest to one end, as many as size, ties
    broken at random."""
    order = np.lexsort((rng.random(len(scores)), -scores if top else scores))
    tail = np.zeros(len(scores), dtype=bool)
    tail[order[:size]] = True
    return tail


@dataclass
class StateSplit:
    """Two groups of a state's members, and what places another member in one."""

    #: groups[side] for side in (False, True); True is the minority.
    groups: Tuple[List[bytes], List[bytes]]
    suffixes: List[bytes]
    #: A member's score is its reads weighed by these; sides places it on the
    #: minority's side at or above cut.
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


def _pvalue(reads, inside) -> np.ndarray:
    """2 min(P(H_v <= h_v), P(H_v >= h_v)) per suffix v, H_v ~ Hypergeometric(n,
    ones of v, |inside|) and h_v the ones of v inside."""
    dist = scipy.stats.hypergeom(len(inside), reads.sum(0), int(inside.sum()))
    hits = reads[inside].sum(0)
    return 2 * np.minimum(dist.cdf(hits), dist.sf(hits - 1))


def _oriented(reads, inside, level):
    """(indices K, orientations o) of the suffixes v with _pvalue <= level /
    #suffixes against inside; o_v = +1 where inside reads above its share."""
    kept = np.flatnonzero(_pvalue(reads, inside) <= level / reads.shape[1])
    above = reads[inside][:, kept].sum(0) >= reads[:, kept].sum(0) * inside.mean()
    return kept, np.where(above, 1, -1)


def _agreeing(reads, kept, signs) -> np.ndarray:
    """Per member, its count of reads over K = kept agreeing with o = signs."""
    return np.where(signs > 0, reads[:, kept], 1 - reads[:, kept]).sum(1)


def _fitted(reads, labels, level):
    """(K, o, k*): K and o by _oriented on labels, and k* midway between the
    mean _agreeing counts of the members labelled in and out; None if K is
    empty."""
    kept, signs = _oriented(reads, labels, level)
    if kept.size == 0:
        return None
    agree = _agreeing(reads, kept, signs)
    return kept, signs, (agree[labels].mean() + agree[~labels].mean()) / 2


def _sharpened(reads, inside, level):
    """The _fitted (K, o, k*) at the fixed point of labelling each member by
    whether its _agreeing count is at least k*, starting from the labels
    inside; None if a step keeps no suffix or no cut separates."""
    seen = set()
    while True:
        fit = _fitted(reads, inside, level)
        if fit is None:
            return None
        labels = _agreeing(reads, *fit[:2]) >= fit[2]
        if labels.all() or not labels.any():
            return None
        if labels.tobytes() in seen:
            return fit
        seen.add(labels.tobytes())
        inside = labels


def split_members(
    draw,
    suffixes,
    placing,
    oracle,
    *,
    minority_below,
    minority_share,
    signal,
    level,
    rng,
):
    """The StateSplit placing n fresh members of draw, n = first_look at the
    level each suffix is tested at, by the _fitted cut over the suffixes marked
    placing for the labels of n others: those _sharpened gives over every
    suffix, starting from the minority_share of the n whose counts over the
    suffixes _going_with picks are lowest (minority_below) or highest, or that
    starting tail where sharpening finds no cut; the smaller side is the
    minority.  None if no suffix marked placing goes with the labels."""
    size = first_look(signal, minority_share, level / len(suffixes), level)
    picking = draw(size)
    reads = _reads(oracle, picking, suffixes)
    empty = np.asarray(oracle.membership_queries(picking), dtype=np.int8)
    picked = _going_with(reads, empty)
    weights = -np.ones(len(picked)) if minority_below else np.ones(len(picked))
    labels = _tail(
        reads[:, picked] @ weights, math.ceil(minority_share * size), rng, True
    )
    sharpened = _sharpened(reads, labels, level)
    if sharpened is not None:
        labels = _agreeing(reads, *sharpened[:2]) >= sharpened[2]
    placeable = np.flatnonzero(placing)
    fit = _fitted(reads[:, placeable], labels, level)
    if fit is None:
        return None
    kept, weights, count = fit
    chosen = [suffixes[placeable[k]] for k in kept]
    # A count of agreeing reads is the weighted reads plus the flipped ones.
    cut = count - (weights < 0).sum()
    testing = draw(size)
    scores = _reads(oracle, testing, chosen) @ weights
    if (scores >= cut).mean() > 1 / 2:
        # The minority is what the cut leaves out; scores are integers.
        weights, scores, cut = -weights, -scores, math.floor(-cut) + 1
    inside = scores >= cut
    return StateSplit(
        groups=(
            [m for m, h in zip(testing, inside) if not h],
            [m for m, h in zip(testing, inside) if h],
        ),
        suffixes=chosen,
        weights=weights,
        cut=cut,
    )


def state_split(pst, dfa, state, family, *, minority_share, level):
    """(split_members of members aimed at state, read on family and fresh
    suffixes and placed by the fresh ones the table does not hold, the aim);
    None where nothing aims at state or split_members finds no split."""
    aim = aim_at(pst, dfa, state)
    if aim is None:
        return None
    fresh = {
        pst.sampler.sample(pst.rng, alphabet_size=pst.alphabet_size)
        for _ in range(fresh_suffixes(level, pst.config.min_suffix_frequency))
    }
    suffixes = [v for v in sorted({pst.table.suffix(v) for v in family} | fresh) if v]
    split = split_members(
        lambda count: [aim() for _ in range(count)],
        suffixes,
        # Members are kept by their noise on the suffixes that place them, so a
        # suffix a later round reads would read them biased.
        [not pst.table.contains_suffix(v) for v in suffixes],
        pst.oracle,
        minority_below=state in dfa.final_states,
        minority_share=minority_share,
        signal=pst.config.min_signal_strength,
        level=level,
        rng=pst.rng,
    )
    return None if split is None else (split, aim)


class SplitSource(RejectionSource):
    """Fresh members of one side of a split: aimed at the split state, kept where
    the split places them on that side.

    Proven on the split's members, which were drawn by the same aim and placed
    the same way, so on_side of drawn is the yield test.  Called dry below the
    rate that count puts a floor under.
    """

    def __init__(self, split, side, aim, oracle, *, on_side, drawn):
        super().__init__()
        assert on_side > 0, "a side nothing lands on has no members to draw"
        self._split = split
        self._side = side
        self._aim = aim
        self._oracle = oracle
        self._poor = float(scipy.stats.beta.ppf(_MISREAD, on_side, drawn - on_side + 1))
        # Not seeded with the split's own members: their reads of the empty suffix
        # decided the split.
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
        return f"SplitSource({'minority' if self._side else 'majority'} side)"
