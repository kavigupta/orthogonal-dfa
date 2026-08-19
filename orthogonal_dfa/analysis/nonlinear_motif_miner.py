"""In-context motif mining on the gate-controlled SpliceAI oracle.

The gate composition-residual oracle is SpliceAI's exon score with a per-length-bin
monotonic-gate bag-of-k-mers prediction (k up to 4) subtracted, removing both first-order
and monotone-nonlinear *composition* -- but not positional structure.

``marginal_motif_stats`` ranks every k-mer (length 1..4) by its *marginal benefit*: the
part of its in-context effect not already explained by its best contained ``(k-1)``-mer.
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
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

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
            residual, flank_l, flank_r,
            [list(m) for m in middles], device=dev, chunk=chunk,
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


def _kmer_effects(score, contexts, k, *, chunk_contexts=200):
    """``(n_contexts, 4**k)`` raw effect ``e = score(insert k-mer at [p, p+k)) - score(bg)``
    for every length-k motif (rows in ``_kmers(k)`` order)."""
    motifs = _kmers(k)
    n = len(motifs)
    rows = []
    for i in range(0, len(contexts), chunk_contexts):
        chunk = contexts[i : i + chunk_contexts]
        base = score([bg for bg, _ in chunk])
        buf = []
        for bg, p in chunk:
            for m in motifs:
                new = list(bg); new[p : p + k] = m; buf.append(new)
        rows.append(score(buf).reshape(len(chunk), n) - base[:, None])
    return motifs, np.concatenate(rows, 0)


@dataclass
class MotifRecord:
    motif: str
    k: int
    marginal: float    # ranking key: effect NOT explained by the best contained (k-1)-mer
    magnitude: float   # raw in-context |rel|, for reference
    n: int

    def __repr__(self):
        return (
            f"{self.motif} [k={self.k}]  marginal={self.marginal:.4f}  "
            f"|effect|={self.magnitude:.4f}"
        )


@permacache("orthogonal_dfa/analysis/nonlinear_motif_miner/marginal_motif_stats_v1")
def marginal_motif_stats(*, n_contexts=3000, max_k=4, edge_margin=0, seed=0):
    """Every k-mer (length 1..``max_k``) ranked by *marginal benefit* -- the part of its
    in-context effect not already explained by its best contained ``(k-1)``-mer.

    Nothing is dropped; longer motifs are included alongside shorter ones and simply sink
    if they add little.  Because backgrounds are uniform-random bases, a contained
    ``(k-1)``-mer's effect is the k-mer effect matrix ``E`` averaged over the free end base
    -- no separate perturbation needed.  With ``E`` reshaped to ``(n_contexts, 4, ..., 4)``,
    dropping the first base is ``mean|E - E.mean(first axis)|`` and dropping the last is
    ``mean|E - E.mean(last axis)|``; the marginal benefit is the smaller residual (the best
    shorter explanation leaves the least unexplained).  For a 1-mer both reduce to the plain
    ``mean|rel|``.  ``magnitude`` (identity-specific ``mean|rel|``) is kept for reference.
    Records are sorted by marginal, descending.  Permacached.
    """
    score = build_controlled_score(default_exon, load_spliceai(400, 0))
    length = default_exon.random_text_length
    pos_range = (
        (edge_margin, length - max_k + 1 - edge_margin) if edge_margin else None
    )
    contexts = sample_contexts(length, max_k, n_contexts, seed=seed, pos_range=pos_range)
    return marginal_motif_records(score, contexts, max_k)


def marginal_motif_records(score, contexts, max_k):
    """The marginal-benefit records for a given ``score`` and context set (see
    :func:`marginal_motif_stats`); factored out so it is testable without the oracle."""
    records = []
    for k in range(1, max_k + 1):
        motifs, E = _kmer_effects(score, contexts, k)
        magnitude = np.abs(E - E.mean(1, keepdims=True)).mean(0)
        T = E.reshape((E.shape[0],) + (4,) * k)
        drop_first = np.abs(T - T.mean(1, keepdims=True)).reshape(E.shape[0], -1).mean(0)
        drop_last = np.abs(T - T.mean(-1, keepdims=True)).reshape(E.shape[0], -1).mean(0)
        marginal = np.minimum(drop_first, drop_last)  # 1-mer: both equal mean|rel|
        for mi, m in enumerate(motifs):
            records.append(MotifRecord(_fmt(m), k, float(marginal[mi]),
                                       float(magnitude[mi]), E.shape[0]))
    records.sort(key=lambda r: -r.marginal)
    return records


# --- sample size for the marginal scan --------------------------------------------


def _maximal_z(n_motifs, delta):
    """Gaussian maximal-bound factor: the max of ``n_motifs`` estimates exceeds the truth
    by at most ``SE * _maximal_z`` with probability >= ``1 - delta``."""
    return math.sqrt(2 * math.log(n_motifs)) + math.sqrt(2 * math.log(1 / delta))


def replicates_for_max_error(residual_std, n_motifs, target_error, *, delta=0.05):
    """Context replicates so the winner's-curse inflation of the *largest* marginal over
    ``n_motifs`` motifs is at most ``target_error`` with probability >= ``1 - delta``.

    Reading the top of a length-k list means selecting the max over ``n_motifs = 4**k``
    marginals, and the maximum of many noisy estimates is biased upward.  Each motif's
    marginal is a context-mean of a residual with per-context standard deviation
    ``residual_std``, so its standard error is ``residual_std / sqrt(n)``.  Treating the
    estimates as approximately normal, the maximum exceeds the truth by at most
    ``SE * (sqrt(2 ln n_motifs) + sqrt(2 ln(1/delta)))`` with probability >= ``1 - delta``
    (a Gaussian maximal bound).  Setting that <= ``target_error`` and solving for ``n``:

        n = ceil( (residual_std * (sqrt(2 ln n_motifs) + sqrt(2 ln(1/delta))) / target_error)^2 )

    Ignoring the positive correlation between overlapping motifs makes this conservative.
    """
    z = _maximal_z(n_motifs, delta)
    return int(math.ceil((residual_std * z / target_error) ** 2))


def marginal_residual_std(score, contexts, k):
    """Conservative per-context std of the marginal residual for length-k motifs: the
    largest, over all length-k motifs and both drop ends, of ``std_ctx|r|`` -- the
    ``residual_std`` that :func:`replicates_for_max_error` needs.  ``contexts`` is a pilot."""
    _, E = _kmer_effects(score, contexts, k)
    T = E.reshape((E.shape[0],) + (4,) * k)
    r_first = np.abs(T - T.mean(1, keepdims=True)).reshape(E.shape[0], -1)
    r_last = np.abs(T - T.mean(-1, keepdims=True)).reshape(E.shape[0], -1)
    return float(np.maximum(r_first.std(0), r_last.std(0)).max())


def replicates_for_marginal_scan(score, contexts, max_k, target_error, *, delta=0.05):
    """For each length k in 1..``max_k``, the context replicates needed to bound the
    winner's-curse error of the top length-k marginal at ``target_error`` (prob >=
    ``1 - delta``), using ``contexts`` as a pilot to estimate the residual std.  Returns
    ``{k: n_replicates}``."""
    return {
        k: replicates_for_max_error(
            marginal_residual_std(score, contexts, k), 4 ** k, target_error, delta=delta
        )
        for k in range(1, max_k + 1)
    }


def _marginal_bound(acc, k, delta):
    """Current winner's-curse bound on the top length-k marginal from streaming sums."""
    n = acc["n"]
    m0, m1 = acc["s0"] / n, acc["s1"] / n
    se0 = np.sqrt(np.maximum(acc["ss0"] / n - m0 ** 2, 0.0) / n)
    se1 = np.sqrt(np.maximum(acc["ss1"] / n - m1 ** 2, 0.0) / n)
    se = np.maximum(se0, se1)  # conservative SE for the min-selected marginal
    return float(se.max() * _maximal_z(4 ** k, delta))


def marginal_records_until(score, make_contexts, max_k, target_error, *,
                           delta=0.05, batch=500, max_contexts=200_000):
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
    acc = {}
    n = 0
    seed = 0
    while True:
        contexts = make_contexts(seed, batch)
        seed += 1
        n += len(contexts)
        for k in range(1, max_k + 1):
            _, E = _kmer_effects(score, contexts, k)
            rel = np.abs(E - E.mean(1, keepdims=True))
            T = E.reshape((E.shape[0],) + (4,) * k)
            rf = np.abs(T - T.mean(1, keepdims=True)).reshape(E.shape[0], -1)
            rl = np.abs(T - T.mean(-1, keepdims=True)).reshape(E.shape[0], -1)
            a = acc.setdefault(
                k, dict(n=0, mag=0.0, s0=0.0, ss0=0.0, s1=0.0, ss1=0.0)
            )
            a["n"] += E.shape[0]
            a["mag"] = a["mag"] + rel.sum(0)
            a["s0"] = a["s0"] + rf.sum(0)
            a["ss0"] = a["ss0"] + (rf ** 2).sum(0)
            a["s1"] = a["s1"] + rl.sum(0)
            a["ss1"] = a["ss1"] + (rl ** 2).sum(0)
        bounds = {k: _marginal_bound(acc[k], k, delta) for k in acc}
        if max(bounds.values()) <= target_error or n >= max_contexts:
            break

    records = []
    for k in range(1, max_k + 1):
        a = acc[k]
        marginal = np.minimum(a["s0"] / a["n"], a["s1"] / a["n"])
        magnitude = a["mag"] / a["n"]
        for mi, m in enumerate(_kmers(k)):
            records.append(MotifRecord(_fmt(m), k, float(marginal[mi]),
                                       float(magnitude[mi]), a["n"]))
    records.sort(key=lambda r: -r.marginal)
    return records, {"n_contexts": n, "bounds": bounds}


@permacache("orthogonal_dfa/analysis/nonlinear_motif_miner/marginal_motif_stats_adaptive_v1")
def marginal_motif_stats_adaptive(*, max_k=4, target_error=0.01, delta=0.05,
                                  batch=500, max_contexts=200_000, edge_margin=0):
    """:func:`marginal_motif_stats` on the gate-controlled SpliceAI-400 oracle, but keep
    sampling fresh contexts until the winner's-curse bound on every length's top marginal
    is <= ``target_error`` (prob >= ``1 - delta``) rather than fixing ``n_contexts``.
    Returns ``(records, info)``.  Permacached."""
    score = build_controlled_score(default_exon, load_spliceai(400, 0))
    length = default_exon.random_text_length
    pos_range = (
        (edge_margin, length - max_k + 1 - edge_margin) if edge_margin else None
    )

    def make_contexts(seed, n):
        return sample_contexts(length, max_k, n, seed=seed, pos_range=pos_range)

    return marginal_records_until(score, make_contexts, max_k, target_error,
                                  delta=delta, batch=batch, max_contexts=max_contexts)
