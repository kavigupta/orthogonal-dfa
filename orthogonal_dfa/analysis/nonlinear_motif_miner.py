"""In-context motif significance on the gate-controlled SpliceAI oracle.

The gate composition-residual oracle (``gate_composition_residual``) is SpliceAI's exon
score with a per-length-bin *monotonic-gate* bag-of-k-mers prediction subtracted, so both
first-order and monotone-nonlinear composition are removed; what remains is positional and
higher-order structure.

``context_motif_significance`` measures, for every k-mer, how much it moves the residual
score *in context*: at each of many consistent-background ``(background, position)``
contexts we overwrite ``[p, p+k)`` with every k-mer and, relative to the average k-mer at
that same spot, take the **magnitude** of the change (``mean|rel|``).  Ranking by magnitude
-- not the signed mean -- surfaces context-dependent, sign-flipping motifs (the reading-
frame stop codons) that a signed average cancels out.

All scoring is batched through ``run_over_middles`` (one GPU pass per chunk).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from orthogonal_dfa.data.exon import RawExon
from orthogonal_dfa.l_star.examples.gate_composition_residual import (
    fit_gate_composition_residual,
)
from orthogonal_dfa.l_star.examples.spliceai_oracle import flanks, run_over_middles
from orthogonal_dfa.spliceai.exon_score import SpliceAIExonScore, device_of

BASES = "ACGT"


def build_controlled_score(
    exon: RawExon,
    spliceai_model,
    *,
    device=None,
    chunk: int = 1024,
    **fit_kw,
) -> Callable[[Sequence[Sequence[int]]], np.ndarray]:
    """A function ``middles -> controlled scores`` for the gate-residual oracle.

    ``middles`` is a list of equal-length int sequences (A=0,C=1,G=2,T=3).  Composition
    is removed by the *monotonic gate* (``fit_gate_composition_residual``), which also
    removes monotone-nonlinear composition, so any motif that survives it is not just
    leftover composition.  The fit is permacached; ``fit_kw`` (``len_lo``, ``len_hi``,
    ``n_max`` ...) forwards on.
    """
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
    """``n_contexts`` ``(background, position)`` pairs -- a consistent background and a
    spot to perturb.  ``position`` is drawn uniformly from ``pos_range`` (default: every
    valid interior position), so aggregating over contexts is position-agnostic."""
    rng = np.random.default_rng(seed)
    lo, hi = pos_range if pos_range is not None else (0, length - motif_k + 1)
    return [
        (rng.integers(0, 4, size=length).tolist(), int(rng.integers(lo, hi)))
        for _ in range(n_contexts)
    ]


@dataclass
class ContextMotifStat:
    motif: str
    magnitude: float   # mean |relative effect| -- effect size REGARDLESS of sign
    mag_z: float       # (magnitude - mean over motifs) / std over motifs -- how much
    #                    LARGER an effect than the alternative k-mers, in magnitude
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
    """For every k-mer, the magnitude of its **in-context** effect vs the alternatives.

    At each context we overwrite ``[p, p+k)`` with every k-mer and record the score
    change ``delta``.  The removed background bases and the position are shared by all
    k-mers at that context, so subtracting the per-context mean over k-mers,
    ``rel = delta - mean_kmers(delta)``, isolates the *identity* of the inserted motif --
    how much more (or less) it moves the score than a typical substitution at that exact
    spot.  ``magnitude = mean|rel|`` is the effect size regardless of sign, and ``mag_z``
    is how many standard deviations that exceeds the average k-mer's -- rank by it, because
    a context-dependent (sign-flipping) motif is invisible to a signed mean.
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
    rel = np.concatenate(rel_rows, axis=0)  # (n_contexts, m_count)

    magnitude = np.abs(rel).mean(0)  # direction-agnostic effect size per motif
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


def _main():
    """Harvest in-context motifs from the gate-controlled SpliceAI-400 oracle:
    perturbations whose effect magnitude, over consistent backgrounds, is larger than
    alternative substitutions at the same spot -- position-agnostic."""
    import os

    from orthogonal_dfa.data.exon import default_exon
    from orthogonal_dfa.spliceai.load_model import load_spliceai

    n_ctx = int(os.environ.get("N_CTX", "3000"))
    motif_k = int(os.environ.get("MOTIF_K", "3"))
    length = default_exon.random_text_length
    # optionally restrict positions to the interior, away from the position-locked
    # donor/acceptor edges, so what surfaces is genuinely context- not edge-driven
    margin = int(os.environ.get("EDGE_MARGIN", "0"))
    pos_range = (margin, length - motif_k + 1 - margin) if margin else None

    print(f"building gate-controlled score (SpliceAI-400)  exon length={length}",
          flush=True)
    score = build_controlled_score(default_exon, load_spliceai(400, 0))

    print(f"in-context harvest: {n_ctx} contexts, {motif_k}-mers, "
          f"pos_range={pos_range or 'all'}", flush=True)
    contexts = sample_contexts(length, motif_k, n_ctx, pos_range=pos_range)
    stats = context_motif_significance(score, contexts, motif_k)

    print("\nmotifs that create the LARGEST in-context effect vs alternatives "
          "(by |effect|, direction-agnostic):")
    for s in sorted(stats, key=lambda s: -s.magnitude)[:15]:
        print("  " + repr(s))


if __name__ == "__main__":
    _main()
