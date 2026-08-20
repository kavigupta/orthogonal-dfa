"""

Contextual motif mining of an oracle.

Every k-mer (sequence of symbols from the underyling language) is
ranked by marginal benefit: the part of its in-context effect not already
explained by its best contained (k-1)-mer.

The marginal effect is defined as follows

substPred(kmer | bg, pos) = prediction(bg[:pos] + kmer + bg[pos+k:])
marginEff(kmer, 'start' | bg, pos) = substPred(kmer | bg, pos) - substPred(kmer[1:] | bg, pos + 1)
marginEff(kmer, 'end' | bg, pos) = substPred(kmer | bg, pos) - substPred(kmer[:-1] | bg, pos)

meanAbsMArgEff(kmer, side) = E_{pos, bg} [ | marginEff(kmer, side | bg, pos) | ]
meanAbsMargEff(kmer) = min_{side} meanAbsMargEff(kmer, side)

The reason for computing a marginal effect is that a contained (k-1)-mer may already explain the effect of a k-mer,
so the marginal effect is a better measure of the k-mer's unique contribution.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np


def sample_contexts(
    length: int,
    n_symbols: int,
    motif_k: int,
    n_contexts: int,
    *,
    seed: int = 0,
    pos_range: Optional[Tuple[int, int]] = None,
) -> List[Tuple[List[int], int]]:
    """n_contexts random (background, position) pairs.

    pos_range defaults to every valid position, so aggregating over contexts is
    position-agnostic.
    """
    rng = np.random.default_rng(seed)
    lo, hi = pos_range if pos_range is not None else (0, length - motif_k + 1)
    return [
        (rng.integers(0, n_symbols, size=length).tolist(), int(rng.integers(lo, hi)))
        for _ in range(n_contexts)
    ]


def _kmers(k, n_symbols):
    return [list(m) for m in itertools.product(range(n_symbols), repeat=k)]


def _fmt(motif, alphabet):
    return "".join(alphabet[c] for c in motif)


def _kmer_effects(score, contexts, k, n_symbols, *, score_batch=100_000):
    """The effect of writing each k-mer into [p, p+k), shaped (n_contexts, n_symbols**k)
    in _kmers order.

    Chunked to keep each score() call near score_batch sequences.
    """
    motifs = _kmers(k, n_symbols)
    n = len(motifs)
    chunk_contexts = max(1, score_batch // n)
    rows = []
    for i in range(0, len(contexts), chunk_contexts):
        chunk = contexts[i : i + chunk_contexts]
        base = score([bg for bg, _ in chunk])
        buf = []
        for bg, p in chunk:
            for m in motifs:
                new = list(bg)
                new[p : p + k] = m
                buf.append(new)
        rows.append(score(buf).reshape(len(chunk), n) - base[:, None])
    return motifs, np.concatenate(rows, 0)


def _marginal_residuals(E, k, n_symbols):
    """rel is the motif's effect against the alternatives at the same spot.

    drop_first and drop_last are what each contained (k-1)-mer leaves unexplained.
    Backgrounds are uniform over the alphabet, so a contained (k-1)-mer's effect is just E
    averaged over the free end symbol.
    """
    T = E.reshape((E.shape[0],) + (n_symbols,) * k)
    rel = np.abs(E - E.mean(1, keepdims=True))
    drop_first = np.abs(T - T.mean(1, keepdims=True)).reshape(E.shape[0], -1)
    drop_last = np.abs(T - T.mean(-1, keepdims=True)).reshape(E.shape[0], -1)
    return rel, drop_first, drop_last


@dataclass
class MotifRecord:
    motif: str
    k: int
    marginal: float  # ranking key: not explained by the best contained (k-1)-mer
    magnitude: float  # in-context |effect|, reported only

    def __repr__(self):
        return (
            f"{self.motif} [k={self.k}]  marginal={self.marginal:.4f}  "
            f"|effect|={self.magnitude:.4f}"
        )


def marginal_motif_records(score, contexts, max_k, alphabet: Sequence[str]):
    """Records over a fixed context set.

    This is the non-streaming reference for marginal_records_until, testable without a
    real oracle.
    """
    records = []
    for k in range(1, max_k + 1):
        motifs, E = _kmer_effects(score, contexts, k, len(alphabet))
        rel, drop_first, drop_last = _marginal_residuals(E, k, len(alphabet))
        marginal = np.minimum(drop_first.mean(0), drop_last.mean(0))
        records += _records_for_length(motifs, k, marginal, rel.mean(0), alphabet)
    records.sort(key=lambda r: -r.marginal)
    return records


def _records_for_length(motifs, k, marginal, magnitude, alphabet):
    return [
        MotifRecord(_fmt(m, alphabet), k, float(marginal[mi]), float(magnitude[mi]))
        for mi, m in enumerate(motifs)
    ]


# --- winner's-curse bound on the top marginal --------------------------------------------


def _maximal_z(n_motifs, delta):
    """Factor by which the max of n_motifs Gaussian estimates can exceed the truth, with
    probability at least 1 - delta."""
    return math.sqrt(2 * math.log(n_motifs)) + math.sqrt(2 * math.log(1 / delta))


@dataclass
class _RunningMoments:
    """Streaming per-motif mean and standard error of a per-context residual."""

    n: int = 0
    total: Any = 0.0
    total_sq: Any = 0.0

    def update(self, residual):
        self.n += residual.shape[0]
        self.total = self.total + residual.sum(0)
        self.total_sq = self.total_sq + (residual**2).sum(0)

    @property
    def mean(self):
        return self.total / self.n

    @property
    def stderr(self):
        var = np.maximum(self.total_sq / self.n - self.mean**2, 0.0)
        return np.sqrt(var / self.n)


@dataclass
class _MarginalAccumulator:
    """Streaming statistics for one motif length."""

    n_symbols: int
    drop_first: _RunningMoments = field(default_factory=_RunningMoments)
    drop_last: _RunningMoments = field(default_factory=_RunningMoments)
    magnitude: _RunningMoments = field(default_factory=_RunningMoments)

    def update(self, E, k):
        rel, drop_first, drop_last = _marginal_residuals(E, k, self.n_symbols)
        self.drop_first.update(drop_first)
        self.drop_last.update(drop_last)
        self.magnitude.update(rel)

    @property
    def marginal(self):
        # the better shorter explanation is the one leaving less unexplained
        return np.minimum(self.drop_first.mean, self.drop_last.mean)

    def bound(self, k, delta):
        """Winner's-curse bound on the top length-k marginal.

        Selecting the max of n_symbols**k estimates inflates it by at most SE * _maximal_z.
        """
        se = np.maximum(self.drop_first.stderr, self.drop_last.stderr)  # conservative
        return float(se.max() * _maximal_z(self.n_symbols**k, delta))


def marginal_records_until(
    score,
    make_contexts,
    max_k,
    target_error,
    alphabet: Sequence[str],
    *,
    delta=0.05,
    contexts_per_round=500,
    max_contexts=200_000,
    on_round=None,
):
    """Stream batches from make_contexts(seed, n) until every length's bound is at most
    target_error, or max_contexts is reached.

    Stopping is on the standard error, not the effect, so the estimates are not biased by
    the stopping rule.
    """
    n_symbols = len(alphabet)
    acc = {k: _MarginalAccumulator(n_symbols) for k in range(1, max_k + 1)}
    n = 0
    seed = 0
    while True:
        contexts = make_contexts(seed, contexts_per_round)
        seed += 1
        n += len(contexts)
        for k, a in acc.items():
            _, E = _kmer_effects(score, contexts, k, n_symbols)
            a.update(E, k)
        bounds = {k: a.bound(k, delta) for k, a in acc.items()}
        if on_round is not None:
            on_round(n, bounds)
        if max(bounds.values()) <= target_error or n >= max_contexts:
            break

    records = []
    for k, a in acc.items():
        records += _records_for_length(
            _kmers(k, n_symbols), k, a.marginal, a.magnitude.mean, alphabet
        )
    records.sort(key=lambda r: -r.marginal)
    return records, {"n_contexts": n, "bounds": bounds}
