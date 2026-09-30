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
from .prefix_sources import aim_at
from .rejection_source import _MISREAD, RejectionSource

#: The largest variance a read can have, at a rate of a half.
_MAX_READ_VARIANCE = 0.25


def tail_ladder(members, minority_share) -> List[int]:
    """[ceil(w n) 2^k for k >= 0, while at most n / 2], w = minority_share,
    n = members."""
    sizes = []
    size = max(1, math.ceil(minority_share * members))
    while size <= members // 2:
        sizes.append(size)
        size *= 2
    return sizes


def fresh_suffixes(miss_rate, preserving_share) -> int:
    """The least m with

        P(Binomial(m, preserving_share) <= 1) <= miss_rate,

    preserving_share being the least class-preserving share of the sampler's
    draws."""
    m = 2
    while scipy.stats.binom.cdf(1, m, preserving_share) > miss_rate:
        m += 1
    return m


def _label_signal(signal, minority_share) -> float:
    """w (1 - w) (2 signal)^2 / v_max, w = minority_share: the squared
    correlation between a member's label and its own read, the labels reading at
    1/2 -+ signal."""
    spread = minority_share * (1 - minority_share)
    return spread * (2 * signal) ** 2 / _MAX_READ_VARIANCE


def first_look(signal, minority_share, level, miss_rate) -> int:
    """ceil((z_level + z_miss)^2 / _label_signal), the members per half at
    which a test on each member's true label would reach power 1 - miss_rate
    at level.  Sizing only, by the normal approximation; the test is exact."""
    z = scipy.stats.norm.isf(level) + scipy.stats.norm.isf(miss_rate)
    return math.ceil(z**2 / _label_signal(signal, minority_share))


def most_looks(signal, minority_share, separating, candidates) -> int:
    """ceil(log2(1 / q)) + 1, with

        q = m^2 (2 signal)^2 w (1 - w) / (M v_max + m^2 (2 signal)^2 w (1 - w))

    the squared correlation with a member's label of its count of ones over all
    M = candidates suffixes, of which m = separating separate."""
    carried = separating**2 * (2 * signal) ** 2 * minority_share * (1 - minority_share)
    quality = carried / (candidates * _MAX_READ_VARIANCE + carried)
    return math.ceil(math.log2(1 / quality)) + 1


def _tail(scores, size, rng, top) -> np.ndarray:
    """A mask of the members ranked furthest to one end, as many as size, ties
    broken at random."""
    order = np.lexsort((rng.random(len(scores)), -scores if top else scores))
    tail = np.zeros(len(scores), dtype=bool)
    tail[order[:size]] = True
    return tail


def _label_pvalue(tail, empty, top) -> float:
    """P(H >= h) if top else P(H <= h), H ~ Hypergeometric(n, sum(E), |T|),
    h = sum over the tail T of E: exact when T is independent of E."""
    dist = scipy.stats.hypergeom(len(tail), int(empty.sum()), int(tail.sum()))
    ones = int(empty[tail].sum())
    return float(dist.sf(ones - 1) if top else dist.cdf(ones))


@dataclass
class StateSplit:
    """Two groups of a state's members, and what places another member in one."""

    #: groups[side] for side in (False, True); True is the minority.
    groups: Tuple[List[bytes], List[bytes]]
    suffixes: List[bytes]
    #: A member's score is its reads weighed by these, and the minority's lie at
    #: or above cut.
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


def _oriented(reads, inside, level, held_out=None):
    """(indices K, orientations o) of the suffixes v with _pvalue <= level /
    #suffixes against inside; o_v = +1 where inside reads above its share.
    held_out = (J, L) tests suffix J[j] against the labels L[:, j] instead."""
    pvalues = _pvalue(reads, inside)
    labels = np.repeat(inside[:, None], reads.shape[1], axis=1)
    if held_out is not None:
        for j, v in enumerate(held_out[0]):
            labels[:, v] = held_out[1][:, j]
            pvalues[v] = _pvalue(reads[:, [v]], labels[:, v])[0]
    kept = np.flatnonzero(pvalues <= level / reads.shape[1])
    hits = np.array([reads[labels[:, v], v].sum() for v in kept])
    shares = np.array([labels[:, v].mean() for v in kept])
    above = hits >= reads[:, kept].sum(0) * shares
    return kept, np.where(above, 1, -1)


def _cut(share, rate_in, rate_out, size) -> float:
    """The least k with w B(k; K, r_in) >= (1 - w) B(k; K, r_out), K = size;
    inf unless 0 < w < 1 and 0 < r_out < r_in < 1."""
    if not (0 < share < 1 and 0 < rate_out < rate_in < 1):
        return math.inf
    slope = math.log(rate_in * (1 - rate_out) / (rate_out * (1 - rate_in)))
    offset = math.log((1 - share) / share) + size * math.log(
        (1 - rate_out) / (1 - rate_in)
    )
    return offset / slope


def _mixture(adjusted, inside, size):
    """(w, r_in, r_out) at the maximum-likelihood fit of

        w Binomial(K, r_in) + (1 - w) Binomial(K, r_out)

    to the counts adjusted, K = size, reached by EM from the share and
    rates of the labels inside and run until the labels _cut gives stop
    changing."""
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
    picking, picked_reads, testing, suffixes, oracle, *, minority_share, level, rng
):
    """(split, p*): p* = min over tail sizes t in tail_ladder and both ends
    of T * _label_pvalue, T the number of such tails, on the testing
    members' counts over the suffixes that _going_with picks on
    picking, whose reads of them are picked_reads.  The split is at the
    least t with T p <= level, or None."""
    members = picking + testing
    empty = np.asarray(oracle.membership_queries(members), dtype=np.int8)
    picks = np.arange(len(members)) < len(picking)
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
    # The tail's share of the testing half, taken of every member.
    inside = _tail(scores, math.ceil(size / len(testing) * len(members)), rng, True)
    cut = scores[inside].min()
    # The tail only has to hold more of the minority than chance to be detected;
    # the sides handed on are placed by every suffix that goes with the label.
    sharpened = _sharpened(
        picked_reads,
        inside[picks],
        _tails_without(picked_reads, picked, weights, inside[picks]),
        level,
    )
    if sharpened is not None:
        kept, weights, count = sharpened
        chosen = [suffixes[k] for k in kept]
        # A count of agreeing reads is the weighted reads plus the flipped ones.
        cut = count - (weights < 0).sum()
        scores[picks] = picked_reads[:, kept] @ weights
        scores[~picks] = _reads(oracle, testing, chosen) @ weights
        if (scores[~picks] >= cut).mean() > 1 / 2:
            # The minority is what the cut leaves out; scores are integers.
            weights, scores, cut = -weights, -scores, math.floor(-cut) + 1
        inside = scores >= cut
    split = StateSplit(
        groups=(
            [m for m, h in zip(testing, inside[~picks]) if not h],
            [m for m, h in zip(testing, inside[~picks]) if h],
        ),
        suffixes=chosen,
        weights=weights,
        cut=cut,
    )
    return split, least


def split_by_looks(
    draw, family, oracle, *, signal, minority_share, preserving_share, alpha, rng
):
    """The split from looks k = 0, 1, ..., L - 1, n_0 2^k members per half with
    n_0 = first_look and L = most_looks, the testing half fresh and the
    picking half every earlier look's topped up with fresh draws, each at level
    alpha / L: the first look's split, or None once look k's p* exceeds
    2^-(k+1).  p* is super-uniform under a pure state, so

        P(look k) <= 2^-k(k+1)/2,   E[members] <= n_0 sum_k 2^k 2^-k(k+1)/2 < 3 n_0."""
    fresh_count = fresh_suffixes(alpha, preserving_share)
    fresh = {draw.suffix() for _ in range(fresh_count)}
    candidates = sorted(set(family) | fresh)
    suffixes = [v for v in candidates if v]
    looks = most_looks(signal, minority_share, 2, len(candidates))
    level = alpha / looks
    size = first_look(signal, minority_share, level, level)
    picking = []
    picked_reads = np.zeros((0, len(suffixes)), dtype=np.int8)
    for look in range(looks):
        # Independent of any later look's testing half, so reused.
        drawn = draw(size - len(picking))
        picking = picking + drawn
        picked_reads = np.concatenate([picked_reads, _reads(oracle, drawn, suffixes)])
        split, p = split_members(
            picking,
            picked_reads,
            draw(size),
            suffixes,
            oracle,
            minority_share=minority_share,
            level=level,
            rng=rng,
        )
        if split is not None:
            return split
        if p > 2 ** -(look + 1):
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


class _Outside:
    """The members from draw that split places on its majority side."""

    def __init__(self, draw, split, oracle):
        self._draw = draw
        self._split = split
        self._oracle = oracle

    def __call__(self, count) -> List[bytes]:
        kept = []
        while len(kept) < count:
            drawn = self._draw(count - len(kept))
            if not drawn:
                break
            sides = self._split.sides(drawn, self._oracle)
            kept += [p for p, side in zip(drawn, sides) if not side]
        return kept

    def suffix(self) -> bytes:
        return self._draw.suffix()


def _minority_bound(split, alpha) -> float:
    """m_hi with P(Binomial(n, m_hi) <= k) = alpha, for k of the split's n
    members on its minority side."""
    minority = len(split.groups[True])
    members = minority + len(split.groups[False])
    if minority == members:
        return 1.0
    return float(scipy.stats.beta.ppf(1 - alpha, minority + 1, members - minority))


def state_split(pst, dfa, state, family, *, alpha):
    """The splits S_1, S_2, ... that split_by_looks finds over prefixes aimed at
    state and the suffixes family, S_j on the members that S_1, ...,
    S_(j-1) all place on their majority side, up to the first S_m with

        share * sum_(j <= m) _minority_bound(S_j) >= merged_minority_mass,

    share the state's mass; and the aim that drew the prefixes.  None if a split
    goes unfound first."""
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
    draw = _Aimed(pst, aim)
    splits = []
    mass = 0.0
    # A minority of several classes is found a class at a time, so a light first
    # one does not show the rest is light too.
    while mass < pst.config.merged_minority_mass:
        found = split_by_looks(
            draw,
            [pst.table.suffix(v) for v in family],
            # Every look reads fresh members, which the memo would only accumulate.
            pst.oracle,
            signal=pst.config.min_signal_strength,
            minority_share=minority_share,
            preserving_share=pst.config.min_suffix_frequency,
            alpha=alpha,
            rng=pst.rng,
        )
        if found is None:
            return None
        splits.append(found)
        mass += share * _minority_bound(found, alpha)
        draw = _Outside(draw, found, pst.oracle)
    return splits, aim


def _placed(splits, prefixes, oracle) -> np.ndarray:
    """For each prefix, the least j with S_j = splits[j] placing it on its
    minority side, or len(splits) if none does."""
    part = np.full(len(prefixes), len(splits))
    pending = np.arange(len(prefixes))
    for j, split in enumerate(splits):
        if not pending.size:
            break
        inside = split.sides([prefixes[i] for i in pending], oracle)
        part[pending[inside]] = j
        pending = pending[~inside]
    return part


class SplitSource(RejectionSource):
    """Fresh members of one part of a chain of splits: aimed at the split state,
    kept where _placed puts them in this part.

    Proven on the first split's members, which were drawn by the same aim and
    placed the same way, so their count in this part is the yield test.  Called
    dry below the rate that count puts a floor under.
    """

    def __init__(self, splits, part, aim, oracle):
        super().__init__()
        self._splits = splits
        self._part = part
        self._aim = aim
        self._oracle = oracle
        first = splits[0]
        drawn = first.groups[True] + first.groups[False]
        on_part = int((_placed(splits, drawn, oracle) == part).sum())
        self._poor = float(
            scipy.stats.beta.ppf(_MISREAD, on_part, len(drawn) - on_part + 1)
        )
        # Not seeded with the split's own members: whichever half they came from,
        # their reads of the empty suffix decided the split.
        self._proven = True

    @property
    def proving(self) -> tuple:
        raise AssertionError("proven on construction; the yield test is never asked")

    @property
    def poor(self) -> float:
        return self._poor

    def attempt_draw(self) -> bool:
        prefix = self._aim()
        if _placed(self._splits, [prefix], self._oracle)[0] != self._part:
            return False
        self._pool.append(prefix)
        return True

    def source_repr(self) -> str:
        return f"SplitSource(part {self._part} of {len(self._splits) + 1})"
