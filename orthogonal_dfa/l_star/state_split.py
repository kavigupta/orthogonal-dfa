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


def _oriented(reads, inside, level, held_out):
    """(indices K, orientations o) of the suffixes v with _pvalue <= level /
    #suffixes against inside; o_v = +1 where inside reads above its share.
    held_out = (J, L) tests suffix J[j] against the labels L[:, j] instead."""
    pvalues = _pvalue(reads, inside)
    labels = np.repeat(inside[:, None], reads.shape[1], axis=1)
    for j, v in enumerate(held_out[0]):
        labels[:, v] = held_out[1][:, j]
        pvalues[v] = _pvalue(reads[:, [v]], labels[:, v])[0]
    kept = np.flatnonzero(pvalues <= level / reads.shape[1])
    hits = np.array([reads[labels[:, v], v].sum() for v in kept])
    shares = np.array([labels[:, v].mean() for v in kept])
    above = hits >= reads[:, kept].sum(0) * shares
    return kept, np.where(above, 1, -1)


def _cut(share, rate_in, rate_out, size) -> float:
    """The k at which w B(k; K, r_in) = (1 - w) B(k; K, r_out), K = size, above
    which the first is the larger; inf unless 0 < w < 1 and 0 < r_out < r_in < 1."""
    if not (0 < share < 1 and 0 < rate_out < rate_in < 1):
        return math.inf
    slope = math.log(rate_in * (1 - rate_out) / (rate_out * (1 - rate_in)))
    offset = math.log((1 - share) / share) + size * math.log(
        (1 - rate_out) / (1 - rate_in)
    )
    return offset / slope


def _mixture(adjusted, inside, size):
    """(w, r_in, r_out) of the classification-EM fit of

        w Binomial(K, r_in) + (1 - w) Binomial(K, r_out)

    to the counts adjusted, K = size, from the share and rates of the labels
    inside, run until the labels _cut gives stop changing."""
    share = inside.mean()
    rate_in = adjusted[inside].mean() / size
    rate_out = adjusted[~inside].mean() / size
    labels = inside
    while not math.isinf(_cut(share, rate_in, rate_out, size)):
        log_in = math.log(share) + scipy.stats.binom.logpmf(adjusted, size, rate_in)
        log_out = math.log(1 - share) + scipy.stats.binom.logpmf(
            adjusted, size, rate_out
        )
        posterior = 1 / (1 + np.exp(log_out - log_in))
        share = posterior.mean()
        rate_in = (posterior * adjusted).sum() / (size * posterior.sum())
        rate_out = ((1 - posterior) * adjusted).sum() / (size * (1 - posterior).sum())
        relabelled = adjusted >= _cut(share, rate_in, rate_out, size)
        if (relabelled == labels).all():
            break
        labels = relabelled
    return share, rate_in, rate_out


def _sharpened(reads, inside, left_out, level):
    """(K, o, k*) at the fixed point of labelling each member by whether its count
    of reads agreeing with o over K is at least k* = _cut of _mixture, K
    and o by _oriented on the labels -- each suffix that set them against the
    labels its own read is left out of, left_out for the first -- starting
    from the labels inside; None if a step keeps no suffix or no cut
    separates."""
    kept, signs = _oriented(reads, inside, level, held_out=left_out)
    seen = set()
    while True:
        if kept.size == 0:
            return None
        agree = np.where(signs > 0, reads[:, kept], 1 - reads[:, kept])
        adjusted = agree.sum(1)
        share, rate_in, rate_out = _mixture(adjusted, inside, len(kept))
        cut = _cut(share, rate_in, rate_out, len(kept))
        labels = adjusted >= cut
        if labels.all() or not labels.any():
            return None
        if labels.tobytes() in seen:
            return kept, signs, cut
        seen.add(labels.tobytes())
        inside = labels
        left_out = adjusted[:, None] - agree >= _cut(
            share, rate_in, rate_out, len(kept) - 1
        )
        kept, signs = _oriented(reads, inside, level, held_out=(kept, left_out))


def _tails_without(reads, picked, weights, tail):
    """(J, L): L[:, j] marks as many members as the tail holds, ranked highest by
    their weighted reads over the suffixes J = picked other than J[j]."""
    scores = reads[:, picked] @ weights
    labels = np.zeros((len(tail), len(picked)), dtype=bool)
    for j, v in enumerate(picked):
        order = np.argsort(-(scores - weights[j] * reads[:, v]), kind="stable")
        labels[order[: int(tail.sum())], j] = True
    return picked, labels


def split_members(
    draw, suffixes, oracle, *, minority_below, minority_share, signal, level, rng
):
    """The StateSplit placing n fresh members of draw, n = first_look at level,
    on each side of the cut _sharpened fits on n others, starting from the
    minority_share of those whose counts over the suffixes _going_with picks
    are lowest (minority_below) or highest; the smaller side is the minority.
    None if sharpening finds no cut."""
    size = first_look(signal, minority_share, level, level)
    picking = draw(size)
    reads = _reads(oracle, picking, suffixes)
    empty = np.asarray(oracle.membership_queries(picking), dtype=np.int8)
    picked = _going_with(reads, empty)
    weights = -np.ones(len(picked)) if minority_below else np.ones(len(picked))
    inside = _tail(
        reads[:, picked] @ weights, math.ceil(minority_share * size), rng, True
    )
    sharpened = _sharpened(
        reads, inside, _tails_without(reads, picked, weights, inside), level
    )
    if sharpened is None:
        return None
    kept, weights, count = sharpened
    chosen = [suffixes[k] for k in kept]
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
    suffixes, the aim); None where nothing aims at state or no cut is found."""
    aim = aim_at(pst, dfa, state)
    if aim is None:
        return None
    fresh = {
        pst.sampler.sample(pst.rng, alphabet_size=pst.alphabet_size)
        for _ in range(fresh_suffixes(level, pst.config.min_suffix_frequency))
    }
    suffixes = sorted({pst.table.suffix(v) for v in family} | fresh)
    split = split_members(
        lambda count: [aim() for _ in range(count)],
        [v for v in suffixes if v],
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
