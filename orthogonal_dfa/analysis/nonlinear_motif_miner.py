"""In-context motif mining on the gate-controlled SpliceAI oracle.

The gate composition-residual oracle is SpliceAI's exon score with a per-length-bin
monotonic-gate bag-of-k-mers prediction (k up to 4) subtracted, removing both first-order
and monotone-nonlinear *composition* -- but not positional structure.

``marginal_motif_stats_adaptive`` ranks every k-mer (length 1..``max_k``) by its *marginal
benefit*: the part of its in-context effect not already explained by its best contained
``(k-1)``-mer.
At many consistent-background ``(background, position)`` contexts we insert each k-mer at
``[p, p+k)``; because backgrounds are uniform-random bases, a contained ``(k-1)``-mer's
effect is just the k-mer effect matrix averaged over the free end base, so the marginal is
the residual after dropping whichever end base matters least.  A longer motif that merely
extends a strong shorter one (``GTAA`` over ``TAA``) adds little and sinks; the reading-
frame stop codons -- positional, so untouched by the count-based control -- rise to the
top.  The table is permacached, so a notebook re-runs instantly.
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
    """``middles -> gate-residual scores`` (A=0,C=1,G=2,T=3).  Composition is removed by
    the monotonic gate (``fit_gate_composition_residual``, permacached); ``fit_kw`` (e.g.
    ``len_lo``, ``len_hi``, ``n_max``) forwards on."""
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
    """``n_contexts`` ``(background, position)`` pairs; ``position`` is uniform over
    ``pos_range`` (default: every valid interior position), so aggregating is
    position-agnostic."""
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
    """``(n_contexts, 4**k)`` raw effect ``e = score(insert k-mer at [p, p+k)) - score(bg)``
    for every length-k motif (rows in ``_kmers(k)`` order).  Chunked so each ``score()``
    call sees about ``max_seqs`` sequences, bounding memory for large k (4**6 = 4096).
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
    """The three per-context residual arrays behind a length-k scan, each shaped like
    ``E`` -- ``(n_contexts, 4**k)``, in ``_kmers(k)`` order.

    ``rel`` is the motif's effect against the alternatives at the same spot.  ``drop_first``
    and ``drop_last`` are what is left after the best contained ``(k-1)``-mer explanation:
    backgrounds are uniform-random bases, so a contained ``(k-1)``-mer's effect is ``E``
    averaged over the free end base, and the residual is the deviation from that average.
    For a 1-mer both reduce to ``rel``.
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
    marginal: float  # ranking key: effect NOT explained by the best contained (k-1)-mer
    magnitude: float  # raw in-context |rel|, for reference

    def __repr__(self):
        return (
            f"{self.motif} [k={self.k}]  marginal={self.marginal:.4f}  "
            f"|effect|={self.magnitude:.4f}"
        )


def marginal_motif_records(score, contexts, max_k):
    """The marginal-benefit records for a given ``score`` and a fixed context set: the
    non-streaming reference for :func:`marginal_records_until`, testable without the
    oracle.

    Nothing is dropped; longer motifs are included alongside shorter ones and simply sink
    if they add little.  Because backgrounds are uniform-random bases, a contained
    ``(k-1)``-mer's effect is the k-mer effect matrix ``E`` averaged over the free end base
    -- no separate perturbation needed.  With ``E`` reshaped to ``(n_contexts, 4, ..., 4)``,
    dropping the first base is ``mean|E - E.mean(first axis)|`` and dropping the last is
    ``mean|E - E.mean(last axis)|``; the marginal benefit is the smaller residual (the best
    shorter explanation leaves the least unexplained).  For a 1-mer both reduce to the plain
    ``mean|rel|``.  ``magnitude`` (identity-specific ``mean|rel|``) is kept for reference.
    Records are sorted by marginal, descending."""
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
    """Gaussian maximal-bound factor: the max of ``n_motifs`` estimates exceeds the truth
    by at most ``SE * _maximal_z`` with probability >= ``1 - delta``."""
    return math.sqrt(2 * math.log(n_motifs)) + math.sqrt(2 * math.log(1 / delta))


@dataclass
class _RunningMoments:
    """Streaming per-motif mean and standard error of a per-context residual.  ``total``
    and ``total_sq`` start as scalar zeros and become ``(4**k,)`` arrays on first update,
    so memory is constant in the number of contexts seen."""

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
    """The streaming statistics for one motif length: the two ``(k-1)``-mer explanations
    whose smaller mean is the marginal benefit, and the magnitude carried for reporting.
    """

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
        """The best shorter explanation leaves the least unexplained."""
        return np.minimum(self.drop_first.mean, self.drop_last.mean)

    def bound(self, k, delta):
        """Current winner's-curse bound on the *top* length-k marginal: reading the top of
        a ``4**k``-long list selects a maximum, which sits above the truth by at most
        ``SE * _maximal_z``.  Both the max over the two drop ends (the min-selected
        marginal is no noisier than the noisier of them) and the max over motifs are
        conservative."""
        se = np.maximum(self.drop_first.stderr, self.drop_last.stderr)
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
    """Stream context batches, accumulating marginal statistics, until the winner's-curse
    bound on every length's top marginal is <= ``target_error`` (prob >= ``1 - delta``), or
    ``max_contexts`` contexts have been used.

    ``make_contexts(seed, n)`` returns ``n`` fresh ``(background, position)`` contexts;
    successive integer seeds give independent batches.  Only per-motif running sums are
    kept, never the full effect matrix, so memory is constant in the total context count.
    Stopping is on the *standard error*, not the effect, so the marginal estimates are not
    biased by the stopping rule.  Returns ``(records, info)`` with
    ``info = {"n_contexts", "bounds": {k: bound}}``.
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
    key_function=dict(
        on_batch=lambda _: None
    ),  # progress callback: not part of the key
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
    """The marginal-benefit ranking on the gate-controlled SpliceAI-400 oracle, sampling
    fresh contexts until the winner's-curse bound on every length's top marginal is
    <= ``target_error`` (prob >= ``1 - delta``) rather than fixing a context count.
    ``on_batch(n, bounds)`` is called after each batch (progress only).  Returns
    ``(records, info)``.  Permacached."""
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
