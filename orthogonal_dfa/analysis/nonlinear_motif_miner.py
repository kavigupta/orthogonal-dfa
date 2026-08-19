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
