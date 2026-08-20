"""In-context motif mining on the gate-controlled SpliceAI oracle.

The oracle is SpliceAI's exon score minus a per-length-bin monotonic-gate bag-of-k-mers
prediction, so composition is removed but positional structure is not.

Every k-mer is ranked by marginal benefit: the part of its in-context effect not already
explained by its best contained (k-1)-mer.

So a longer motif that merely extends a strong shorter one, GTAA over TAA, adds little
and sinks.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
from permacache import permacache

from orthogonal_dfa.data.exon import RawExon, default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import (
    fit_gate_composition_residual,
)
from orthogonal_dfa.l_star.examples.spliceai_oracle import flanks, run_over_middles
from orthogonal_dfa.spliceai.exon_score import SpliceAIExonScore, device_of
from orthogonal_dfa.spliceai.load_model import load_spliceai

BASES = "ACGT"


def build_controlled_score(
    exon: RawExon,
    spliceai_model,
    *,
    device=None,
    chunk: int = 1024,
    **fit_kw,
) -> Callable[[Sequence[Sequence[int]]], np.ndarray]:
    """Maps middles, over the alphabet A=0 C=1 G=2 T=3, to gate-residual scores.

    fit_kw forwards to the gate fit: len_lo, len_hi, n_max.
    """
    score_model = SpliceAIExonScore(spliceai_model).eval()
    residual = fit_gate_composition_residual(score_model, exon, device=device, **fit_kw)
    dev = device_of(score_model, device)
    flank_l, flank_r = flanks(exon)

    def score(middles: Sequence[Sequence[int]]) -> np.ndarray:
        return run_over_middles(
            residual,
            flank_l,
            flank_r,
            [list(m) for m in middles],
            device=dev,
            chunk=chunk,
        )

    return score


def sample_contexts(
    length: int,
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
        (rng.integers(0, 4, size=length).tolist(), int(rng.integers(lo, hi)))
        for _ in range(n_contexts)
    ]


# --- marginal-benefit ranking across motif lengths --------------------------------


def _kmers(k):
    return [list(m) for m in itertools.product(range(4), repeat=k)]


def _fmt(motif):
    return "".join(BASES[c] for c in motif)


def _kmer_effects(score, contexts, k, *, max_seqs=100_000):
    """The effect of writing each k-mer into [p, p+k), shaped (n_contexts, 4**k) in
    _kmers(k) order.

    Chunked to keep each score() call near max_seqs sequences.
    """
    motifs = _kmers(k)
    n = len(motifs)
    chunk_contexts = max(1, max_seqs // n)
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


def _marginal_residuals(E, k):
    """rel is the motif's effect against the alternatives at the same spot.

    drop_first and drop_last are what each contained (k-1)-mer leaves unexplained.
    Backgrounds are uniform-random bases, so a contained (k-1)-mer's effect is just E
    averaged over the free end base.
    """
    T = E.reshape((E.shape[0],) + (4,) * k)
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


def marginal_motif_records(score, contexts, max_k):
    """Records over a fixed context set.

    This is the non-streaming reference for marginal_records_until, testable without the
    oracle.
    """
    records = []
    for k in range(1, max_k + 1):
        motifs, E = _kmer_effects(score, contexts, k)
        rel, drop_first, drop_last = _marginal_residuals(E, k)
        marginal = np.minimum(drop_first.mean(0), drop_last.mean(0))
        records += _records_for_length(motifs, k, marginal, rel.mean(0))
    records.sort(key=lambda r: -r.marginal)
    return records


def _records_for_length(motifs, k, marginal, magnitude):
    return [
        MotifRecord(_fmt(m), k, float(marginal[mi]), float(magnitude[mi]))
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

    drop_first: _RunningMoments = field(default_factory=_RunningMoments)
    drop_last: _RunningMoments = field(default_factory=_RunningMoments)
    magnitude: _RunningMoments = field(default_factory=_RunningMoments)

    def update(self, E, k):
        rel, drop_first, drop_last = _marginal_residuals(E, k)
        self.drop_first.update(drop_first)
        self.drop_last.update(drop_last)
        self.magnitude.update(rel)

    @property
    def marginal(self):
        # the better shorter explanation is the one leaving less unexplained
        return np.minimum(self.drop_first.mean, self.drop_last.mean)

    def bound(self, k, delta):
        """Winner's-curse bound on the top length-k marginal.

        Selecting the max of 4**k estimates inflates it by at most SE * _maximal_z.
        """
        se = np.maximum(self.drop_first.stderr, self.drop_last.stderr)  # conservative
        return float(se.max() * _maximal_z(4**k, delta))


def marginal_records_until(
    score,
    make_contexts,
    max_k,
    target_error,
    *,
    delta=0.05,
    batch=500,
    max_contexts=200_000,
    on_batch=None,
):
    """Stream batches from make_contexts(seed, n) until every length's bound is at most
    target_error, or max_contexts is reached.

    Stopping is on the standard error, not the effect, so the estimates are not biased by
    the stopping rule.
    """
    acc = {k: _MarginalAccumulator() for k in range(1, max_k + 1)}
    n = 0
    seed = 0
    while True:
        contexts = make_contexts(seed, batch)
        seed += 1
        n += len(contexts)
        for k, a in acc.items():
            _, E = _kmer_effects(score, contexts, k)
            a.update(E, k)
        bounds = {k: a.bound(k, delta) for k, a in acc.items()}
        if on_batch is not None:
            on_batch(n, bounds)
        if max(bounds.values()) <= target_error or n >= max_contexts:
            break

    records = []
    for k, a in acc.items():
        records += _records_for_length(_kmers(k), k, a.marginal, a.magnitude.mean)
    records.sort(key=lambda r: -r.marginal)
    return records, {"n_contexts": n, "bounds": bounds}


@permacache(
    "orthogonal_dfa/analysis/nonlinear_motif_miner/marginal_motif_stats_adaptive_v1",
    key_function=dict(on_batch=lambda _: None),  # progress only: not part of the key
)
def marginal_motif_stats_adaptive(
    *,
    max_k=4,
    target_error=0.01,
    delta=0.05,
    batch=500,
    max_contexts=200_000,
    edge_margin=0,
    on_batch=None,
):
    """marginal_records_until on the gate-controlled SpliceAI-400 oracle."""
    score = build_controlled_score(default_exon, load_spliceai(400, 0))
    length = default_exon.random_text_length
    pos_range = (edge_margin, length - max_k + 1 - edge_margin) if edge_margin else None

    def make_contexts(seed, n):
        return sample_contexts(length, max_k, n, seed=seed, pos_range=pos_range)

    return marginal_records_until(
        score,
        make_contexts,
        max_k,
        target_error,
        delta=delta,
        batch=batch,
        max_contexts=max_contexts,
        on_batch=on_batch,
    )
