"""In-context motif significance on the gate-controlled SpliceAI oracle.

The gate composition-residual oracle is SpliceAI's exon score with a per-length-bin
monotonic-gate bag-of-k-mers prediction subtracted, removing both first-order and
monotone-nonlinear composition.  For every k-mer we measure the *magnitude* of its
in-context effect: at many consistent-background ``(background, position)`` contexts we
overwrite ``[p, p+k)`` with every k-mer and, relative to the average k-mer at that spot,
take ``mean|rel|``.  Ranking by magnitude rather than the signed mean surfaces context-
dependent, sign-flipping motifs -- the reading-frame stop codons -- that a signed average
cancels out.  The top-motif table is permacached, so a notebook re-runs instantly.
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


@dataclass
class ContextMotifStat:
    """A k-mer's in-context effect size vs the alternatives at the same spot."""

    motif: str
    magnitude: float  # mean|rel|: effect size regardless of sign
    mag_z: float      # SDs by which magnitude exceeds the average k-mer's
    n: int

    def __repr__(self):
        return (
            f"{self.motif}  |effect|={self.magnitude:.4f} (mag_z={self.mag_z:+5.1f})  "
            f"n={self.n}"
        )


def context_motif_significance(
    score: Callable,
    contexts: List[Tuple[List[int], int]],
    motif_k: int,
    *,
    chunk_contexts: int = 200,
) -> List[ContextMotifStat]:
    """Each k-mer's in-context effect magnitude vs the alternatives.

    At each context we overwrite ``[p, p+k)`` with every k-mer; subtracting the per-context
    mean over k-mers, ``rel = delta - mean_kmers(delta)``, isolates the inserted motif's
    identity (controlling for the removed bases and the position).  ``magnitude = mean|rel|``
    is the sign-agnostic effect size and ``mag_z`` its z-score across motifs -- rank by it,
    since a sign-flipping motif is invisible to a signed mean.
    """
    motifs = [list(m) for m in itertools.product(range(4), repeat=motif_k)]
    m_count = len(motifs)
    rel_rows: List[np.ndarray] = []
    for i in range(0, len(contexts), chunk_contexts):
        chunk = contexts[i : i + chunk_contexts]
        base = score([bg for bg, _ in chunk])
        perturbed: List[List[int]] = []
        for bg, p in chunk:
            for m in motifs:
                new = list(bg)
                new[p : p + motif_k] = m
                perturbed.append(new)
        delta = score(perturbed).reshape(len(chunk), m_count) - base[:, None]
        rel_rows.append(delta - delta.mean(1, keepdims=True))
    rel = np.concatenate(rel_rows, axis=0)

    magnitude = np.abs(rel).mean(0)
    mag_z = (magnitude - magnitude.mean()) / magnitude.std()
    return [
        ContextMotifStat(
            motif="".join(BASES[c] for c in m),
            magnitude=float(magnitude[i]),
            mag_z=float(mag_z[i]),
            n=rel.shape[0],
        )
        for i, m in enumerate(motifs)
    ]


@permacache("orthogonal_dfa/analysis/nonlinear_motif_miner/context_motif_stats_v1")
def context_motif_stats(*, n_contexts=3000, motif_k=3, edge_margin=0, seed=0):
    """Every k-mer's in-context stats on the gate-controlled SpliceAI-400 oracle, sorted by
    magnitude (largest effect vs alternatives first).  Permacached.

    ``edge_margin`` drops positions within that many nt of either exon edge (excludes the
    position-locked donor/acceptor edges to confirm a signal is context- not edge-driven).
    """
    score = build_controlled_score(default_exon, load_spliceai(400, 0))
    length = default_exon.random_text_length
    pos_range = (
        (edge_margin, length - motif_k + 1 - edge_margin) if edge_margin else None
    )
    contexts = sample_contexts(length, motif_k, n_contexts, seed=seed, pos_range=pos_range)
    stats = context_motif_significance(score, contexts, motif_k)
    return sorted(stats, key=lambda s: -s.magnitude)


def plot_top_motifs(stats, *, top=15, ax=None):
    """Horizontal bar chart of the top motifs by in-context effect magnitude."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    top_stats = sorted(stats, key=lambda s: -s.magnitude)[:top]
    ax.barh(range(len(top_stats)), [s.magnitude for s in top_stats], color="#4c72b0")
    ax.set_yticks(range(len(top_stats)))
    ax.set_yticklabels([s.motif for s in top_stats])
    ax.invert_yaxis()
    ax.set_xlabel("in-context effect magnitude  |rel|")
    return ax
