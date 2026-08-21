"""

Contextual motif mining of an oracle.

Every k-mer (sequence of symbols from the underyling language) is
ranked by marginal effect: the part of its in-context effect not already
explained by its best contained (k-1)-mer.

The marginal effect is defined as follows

substPred(kmer | bg, pos) = prediction(bg[:pos] + kmer + bg[pos+k:])
absMarginEff(kmer, 'start' | bg, pos) = | substPred(kmer | bg, pos) - E_s substPred(s + kmer[1:] | bg, pos) |
absMarginEff(kmer, 'end' | bg, pos) = | substPred(kmer | bg, pos) - E_s substPred(kmer[:-1] + s | bg, pos) |

where s is a uniformly random symbol, so the subtracted term is the (k-1)-mer's effect
averaged over the dropped end.

meanAbsMArgEff(kmer, side) = E_{pos, bg} [ absMarginEff(kmer, side | bg, pos) ]
meanAbsMargEff(kmer) = min_{side} meanAbsMargEff(kmer, side)

The reason for computing a marginal effect is that a contained (k-1)-mer may already explain the effect of a k-mer,
so the marginal effect is a better measure of the k-mer's unique contribution.
"""

from __future__ import annotations

import itertools
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Sequence, Tuple

import numpy as np
import tqdm.auto as tqdm
from permacache import permacache


class ScoreOracle(ABC):
    """What the miner mines: something that scores sequences and knows its own shape."""

    @property
    @abstractmethod
    def alphabet(self) -> Sequence[str]:
        """The symbols, indexed by the integers a sequence is written in."""

    @property
    @abstractmethod
    def length(self) -> int:
        """Length of the sequences this scores."""

    @abstractmethod
    def scores(self, seqs) -> np.ndarray:
        """(n_seqs, length) of symbol indices -> (n_seqs,) of scores."""

    @abstractmethod
    def hash_payload(self):
        """Stable identity of this oracle, distinguishing instances of the same class."""

    def __permacache_hash__(self):
        # permacache does not mix the class into a custom hash, so we do it here
        return [
            type(self).__module__ + "." + type(self).__name__,
            self.hash_payload(),
        ]

    @property
    def n_symbols(self) -> int:
        return len(self.alphabet)


def sample_contexts(
    length: int,
    n_symbols: int,
    motif_k: int,
    n_contexts: int,
    *,
    seed=0,
) -> List[Tuple[List[int], int]]:
    rng = np.random.default_rng(seed)
    return [
        (
            rng.integers(0, n_symbols, size=length).tolist(),
            int(rng.integers(0, length - motif_k + 1)),
        )
        for _ in range(n_contexts)
    ]


def _kmers(k, n_symbols):
    return np.array(list(itertools.product(range(n_symbols), repeat=k)))


def _fmt(motif, alphabet):
    return "".join(alphabet[c] for c in motif)


def _kmer_scores(oracle, contexts, k, *, score_batch=100_000):
    """
    Output is (motifs, E) where E[i, j] is the score of motif j in context i.

    """
    motifs = _kmers(k, oracle.n_symbols)
    n = len(motifs)
    chunk_contexts = max(1, score_batch // n)
    rows = []
    for i in tqdm.trange(0, len(contexts), chunk_contexts):
        chunk = contexts[i : i + chunk_contexts]
        backgrounds = np.array([bg for bg, _ in chunk])
        # seqs[context, motif] = sequence
        seqs = np.repeat(backgrounds[:, None, :], n, axis=1)
        for ci, (_, p) in enumerate(chunk):
            seqs[ci, :, p : p + k] = motifs
        flat = seqs.reshape(len(chunk) * n, -1)
        rows.append(oracle.scores(flat).reshape(len(chunk), n))
    return motifs, np.concatenate(rows, 0)


def _abs_marginal_residuals(E, k, n_symbols):
    """
    E: shape (n_contexts, n_motifs) array of scores for each motif in each context
    k: length of the motifs
    n_symbols: size of the alphabet

    Returns (drop_first, drop_last) where
        drop_side[bg_pos_idx, motif_idx] represents absMarginEff(kmer, side | bg, pos)
    """

    T = E.reshape((E.shape[0],) + (n_symbols,) * k)
    drop_first = np.abs(T - T.mean(1, keepdims=True)).reshape(E.shape[0], -1)
    drop_last = np.abs(T - T.mean(-1, keepdims=True)).reshape(E.shape[0], -1)
    return drop_first, drop_last


def _uniform_offsets(positions, length, k, max_k, rng):
    """
    Where inside each context's max_k window to read the length-k motif from.
    """
    n_positions = length - max_k + 1
    n_starts = length - k + 1
    # pos -> (pos * n_starts + offset) // n_positions; automatically rescales it
    spread = positions * n_starts + rng.integers(0, n_starts, positions.shape)
    return spread // n_positions - positions


def _restrict_to_length(E, contexts, k, max_k, n_symbols, *, rng):
    """The length-k table, read out of the length-max_k table E.

    The motif is the window at the offset _uniform_offsets picks; the symbols on either
    side of it are drawn, which is the same as reading what the background had, since the
    background is uniform there.  _kmers is lexicographic, so the window's columns are
    (prefix, motif, suffix) in that nesting.
    """
    d = max_k - k
    if not d:
        return E
    length = len(contexts[0][0])
    offsets = _uniform_offsets(
        np.array([p for _, p in contexts]), length, k, max_k, rng
    )
    out = np.empty((E.shape[0], n_symbols**k), dtype=E.dtype)
    for offset in range(d + 1):
        rows = np.flatnonzero(offsets == offset)
        if rows.size == 0:
            continue
        before, after = n_symbols**offset, n_symbols ** (d - offset)
        window = E[rows].reshape(len(rows), before, n_symbols**k, after)
        out[rows] = window[
            np.arange(len(rows)),
            rng.integers(0, before, size=len(rows)),
            :,
            rng.integers(0, after, size=len(rows)),
        ]
    return out


def _abs_marginal_residuals_all(E, contexts, max_k, oracle, rng):
    """_abs_marginal_residuals for every length 1..max_k, concatenated along the motif
    axis in _all_motifs order, so one E covers every length."""
    sides = [
        _abs_marginal_residuals(
            _restrict_to_length(E, contexts, k, max_k, oracle.n_symbols, rng=rng),
            k,
            oracle.n_symbols,
        )
        for k in range(1, max_k + 1)
    ]
    return tuple(np.concatenate(side, 1) for side in zip(*sides))


def _all_motifs(max_k, alphabet):
    """Every motif of length 1..max_k, in the order the concatenated motif axis uses."""
    return [
        _fmt(m, alphabet) for k in range(1, max_k + 1) for m in _kmers(k, len(alphabet))
    ]


def _n_motifs(max_k, n_symbols):
    """How many motifs the concatenated motif axis ranks against each other."""
    return sum(n_symbols**k for k in range(1, max_k + 1))


def _maximal_z(n_motifs, delta):
    """Factor by which the max of n_motifs Gaussian estimates can exceed the truth, with
    probability at least 1 - delta."""
    return math.sqrt(2 * math.log(n_motifs)) + math.sqrt(2 * math.log(1 / delta))


@dataclass
class MotifRecord:
    motif: str
    marginal: float  # ranking key: not explained by the best contained (k-1)-mer

    @property
    def k(self):
        return len(self.motif)

    def __repr__(self):
        return f"{self.motif} [k={self.k}]  marginal={self.marginal:.4f}"


@dataclass
class _RunningMoments:
    """Streaming per-motif mean and standard error of a per-context residual."""

    n: int = 0
    total: Any = 0.0
    total_sq: Any = 0.0

    @classmethod
    def of(cls, residual):
        return cls(residual.shape[0], residual.sum(0), (residual**2).sum(0))

    def __add__(self, other):
        return _RunningMoments(
            self.n + other.n, self.total + other.total, self.total_sq + other.total_sq
        )

    @property
    def mean(self):
        return self.total / self.n

    @property
    def stderr(self):
        var = np.maximum(self.total_sq / self.n - self.mean**2, 0.0)
        return np.sqrt(var / self.n)


@dataclass
class _MarginalAccumulator:
    """Streaming statistics for every length at once, over the concatenated motif axis."""

    n_symbols: int
    max_k: int
    drop_first: _RunningMoments = field(default_factory=_RunningMoments)
    drop_last: _RunningMoments = field(default_factory=_RunningMoments)

    @classmethod
    def of(cls, E, contexts, oracle, max_k, *, rng):
        drop_first, drop_last = _abs_marginal_residuals_all(
            E, contexts, max_k, oracle, rng
        )
        return cls(
            oracle.n_symbols,
            max_k,
            _RunningMoments.of(drop_first),
            _RunningMoments.of(drop_last),
        )

    def __add__(self, other):
        assert (self.n_symbols, self.max_k) == (other.n_symbols, other.max_k)
        return _MarginalAccumulator(
            self.n_symbols,
            self.max_k,
            self.drop_first + other.drop_first,
            self.drop_last + other.drop_last,
        )

    @property
    def n_contexts(self):
        return self.drop_first.n

    @property
    def marginal(self):
        # the better shorter explanation is the one leaving less unexplained
        return np.minimum(self.drop_first.mean, self.drop_last.mean)

    def bound(self, delta):
        """Winner's-curse bound on the top marginal.

        The top motif is picked across every length at once, so the selection is over all
        _n_motifs of them, and being the max inflates the estimate by at most

            SE * _maximal_z
        """
        se = np.maximum(self.drop_first.stderr, self.drop_last.stderr)  # conservative
        z = _maximal_z(_n_motifs(self.max_k, self.n_symbols), delta)
        return float(se.max() * z)


@permacache("orthogonal_dfa/analysis/nonlinear_motif_miner/round_accumulator_v2")
def _round_accumulator(oracle, max_k, contexts_per_round, seed, round_index):
    """One round's statistics, cached on its own so a rerun with a tighter target_error
    reuses every round it already paid for instead of starting over."""
    contexts = sample_contexts(
        oracle.length,
        oracle.n_symbols,
        max_k,
        contexts_per_round,
        seed=[seed, 0, round_index],
    )
    _, E = _kmer_scores(oracle, contexts, max_k)
    rng = np.random.default_rng([seed, 1, round_index])
    return _MarginalAccumulator.of(E, contexts, oracle, max_k, rng=rng)


def marginal_records_until(
    oracle: ScoreOracle,
    max_k,
    target_error,
    *,
    delta=0.05,
    contexts_per_round=500,
    max_contexts=200_000,
    seed=0,
):
    """Draw rounds of contexts until the winner-curse bound on the top marginal is at most
    target_error, or max_contexts is reached.

    Stopping is on the standard error, not the effect, so the estimates are not biased by
    the stopping rule.  info["bound"] is what was achieved, which differs from target_error
    only when max_contexts cut the scan short.
    """
    acc = _MarginalAccumulator(oracle.n_symbols, max_k)
    round_index = 0
    # delay so short scans (tests, small oracles) stay silent
    with tqdm.tqdm(total=max_contexts, unit="ctx", delay=5) as pbar:
        while True:
            acc = acc + _round_accumulator(
                oracle, max_k, contexts_per_round, seed, round_index
            )
            round_index += 1
            bound = acc.bound(delta)
            pbar.n = acc.n_contexts
            pbar.set_postfix(bound=f"{bound:.4f}")
            if bound <= target_error or acc.n_contexts >= max_contexts:
                break

    records = [
        MotifRecord(motif, float(v))
        for motif, v in zip(
            _all_motifs(max_k, oracle.alphabet), acc.marginal, strict=True
        )
    ]
    records.sort(key=lambda r: -r.marginal)
    return records, {"n_contexts": acc.n_contexts, "bound": bound}
