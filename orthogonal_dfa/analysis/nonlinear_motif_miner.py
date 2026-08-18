"""Nonlinear motif harvesting on the linearly-controlled SpliceAI oracle.

The linear composition-residual oracle (``composition_residual``) is SpliceAI's exon
score with a per-length-bin *linear* bag-of-k-mers prediction subtracted.  First-order
composition is therefore removed; what remains is positional and higher-order structure.
This module finds that structure by **perturbation analysis** -- in-silico mutagenesis
plus motif-insertion scans -- and, crucially, separates genuinely **nonlinear** motifs
from leftover additive (composition-like) effects via an epistasis test.

The pipeline (``harvest``):

1. ``single_base_ism`` -- for every (position, base) substitution, the mean change in
   the controlled score over a background sample: an ``(L, 4)`` saliency map.
2. Pick the highest-saliency positions.
3. ``motif_effects`` -- insert every k-mer at those positions; measure the mean score
   change.  Perturbations with large effects are the candidate motifs.
4. ``nonlinear_component`` -- each motif's observed effect minus the sum of its single-
   base ISM effects.  A motif whose effect equals the additive prediction is an additive
   (composition-like, what the *gate* would also remove) effect; a large residual is a
   genuine nonlinear/epistatic motif -- the harvest target.

All scoring is batched through ``run_over_middles`` (one GPU pass per chunk).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from orthogonal_dfa.data.exon import RawExon
from orthogonal_dfa.l_star.examples.composition_residual import fit_composition_residual
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
    """A function ``middles -> controlled scores`` for the *linear* residual oracle.

    ``middles`` is a list of equal-length int sequences (A=0,C=1,G=2,T=3).  The residual
    fit is permacached; ``fit_kw`` (``len_lo``, ``len_hi``, ``n_max`` ...) forwards to
    ``fit_composition_residual``.
    """
    score_model = SpliceAIExonScore(spliceai_model).eval()
    residual = fit_composition_residual(score_model, exon, device=device, **fit_kw)
    dev = device_of(score_model, device)
    flank_l, flank_r = flanks(exon)

    def score(middles: Sequence[Sequence[int]]) -> np.ndarray:
        return run_over_middles(
            residual, flank_l, flank_r,
            [list(m) for m in middles], device=dev, chunk=chunk,
        )

    return score


def sample_backgrounds(length: int, n: int, seed: int = 0) -> List[List[int]]:
    """``n`` random background middles of the given length."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 4, size=length).tolist() for _ in range(n)]


def _score_perturbations(
    score: Callable, backgrounds: List[List[int]], perturb, jobs
) -> np.ndarray:
    """Mean over backgrounds of ``score(perturbed) - score(background)`` for each job.

    ``jobs`` is a list of specs; ``perturb(background, spec) -> new_middle``.  Only
    backgrounds actually changed by a job contribute (a no-op substitution has effect 0
    and is skipped), and the divisor is the number of backgrounds, so a job that rarely
    changes anything gets a correspondingly small mean.  One batched score() call.
    """
    base = score(backgrounds)
    n = len(backgrounds)
    perturbed: List[List[int]] = []
    index: List[Tuple[int, int]] = []  # (job_id, background_id)
    for j, spec in enumerate(jobs):
        for b, bg in enumerate(backgrounds):
            new = perturb(bg, spec)
            if new is not None:
                perturbed.append(new)
                index.append((j, b))
    if not perturbed:
        return np.zeros(len(jobs))
    delta = score(perturbed) - base[[b for _, b in index]]
    out = np.zeros(len(jobs))
    counts = np.zeros(len(jobs))  # kept for reference; divisor is n (population mean)
    for (j, _), d in zip(index, delta):
        out[j] += d
        counts[j] += 1
    return out / n


def single_base_ism(
    score: Callable, backgrounds: List[List[int]]
) -> np.ndarray:
    """``(L, 4)`` saliency: mean change in the controlled score from forcing each base
    at each position (in-silico mutagenesis)."""
    L = len(backgrounds[0])
    jobs = [(i, b) for i in range(L) for b in range(4)]

    def perturb(bg, spec):
        i, b = spec
        if bg[i] == b:
            return None
        new = list(bg)
        new[i] = b
        return new

    flat = _score_perturbations(score, backgrounds, perturb, jobs)
    return flat.reshape(L, 4)


def motif_effects(
    score: Callable,
    backgrounds: List[List[int]],
    motifs: Sequence[Sequence[int]],
    positions: Sequence[int],
) -> np.ndarray:
    """``(len(motifs), len(positions))`` mean score change from substituting each motif
    at each position (the motif overwrites ``[p, p+len(motif))``)."""
    L = len(backgrounds[0])
    jobs = [(tuple(m), p) for m in motifs for p in positions]

    def perturb(bg, spec):
        m, p = spec
        if p + len(m) > L:
            return None
        new = list(bg)
        new[p : p + len(m)] = m
        return new

    flat = _score_perturbations(score, backgrounds, perturb, jobs)
    return flat.reshape(len(motifs), len(positions))


def additive_prediction(
    ism: np.ndarray, motif: Sequence[int], position: int
) -> float:
    """The sum of the single-base ISM effects for a motif at a position -- what the
    motif's effect would be if its bases acted independently."""
    return float(sum(ism[position + j, motif[j]] for j in range(len(motif))))


@dataclass
class MotifHit:
    motif: str
    position: int
    effect: float          # observed mean score change
    additive: float        # sum of single-base ISM effects
    nonlinear: float       # effect - additive (epistasis)

    def __repr__(self):
        return (
            f"{self.motif}@{self.position:>3}  effect={self.effect:+.3f}  "
            f"additive={self.additive:+.3f}  nonlinear={self.nonlinear:+.3f}"
        )


def harvest(
    score: Callable,
    length: int,
    *,
    n_backgrounds: int = 200,
    motif_k: int = 3,
    top_positions: int = 20,
    seed: int = 0,
) -> Tuple[np.ndarray, List[MotifHit]]:
    """Run the full pipeline and return ``(ism_saliency, hits)``.

    ``hits`` are every (k-mer, high-saliency position) pair with their observed effect,
    additive prediction and nonlinear (epistatic) component -- unsorted; sort by
    ``abs(h.nonlinear)`` for the nonlinear harvest or ``abs(h.effect)`` for the strongest
    perturbations.
    """
    backgrounds = sample_backgrounds(length, n_backgrounds, seed=seed)
    ism = single_base_ism(score, backgrounds)

    # positions where mutagenesis moves the score most (max range over bases)
    saliency = ism.max(1) - ism.min(1)
    positions = sorted(int(p) for p in np.argsort(-saliency)[:top_positions]
                       if p + motif_k <= length)

    motifs = [list(m) for m in itertools.product(range(4), repeat=motif_k)]
    eff = motif_effects(score, backgrounds, motifs, positions)

    hits: List[MotifHit] = []
    for mi, m in enumerate(motifs):
        for pj, p in enumerate(positions):
            add = additive_prediction(ism, m, p)
            observed = float(eff[mi, pj])
            hits.append(MotifHit(
                motif="".join(BASES[c] for c in m),
                position=p,
                effect=observed,
                additive=add,
                nonlinear=observed - add,
            ))
    return ism, hits


def _main():
    """Harvest nonlinear motifs from the linear-controlled SpliceAI-400 oracle."""
    import os

    from orthogonal_dfa.data.exon import default_exon
    from orthogonal_dfa.spliceai.load_model import load_spliceai

    n_bg = int(os.environ.get("N_BG", "200"))
    motif_k = int(os.environ.get("MOTIF_K", "3"))
    top_positions = int(os.environ.get("TOP_POS", "20"))
    length = default_exon.random_text_length

    print(f"building linear-controlled score (SpliceAI-400)  length={length}", flush=True)
    score = build_controlled_score(default_exon, load_spliceai(400, 0))

    print(f"harvesting: {n_bg} backgrounds, {motif_k}-mers, top {top_positions} positions",
          flush=True)
    ism, hits = harvest(score, length, n_backgrounds=n_bg, motif_k=motif_k,
                        top_positions=top_positions)

    saliency = ism.max(1) - ism.min(1)
    print("\ntop-10 saliency positions (in-silico mutagenesis range):")
    for p in np.argsort(-saliency)[:10]:
        dom = BASES[int(ism[p].argmax())]
        print(f"  pos {int(p):>3}: range={saliency[p]:.3f}  favours {dom}")

    print("\ntop-15 NONLINEAR (epistatic) motifs -- effect exceeds the additive prediction:")
    for h in sorted(hits, key=lambda h: -abs(h.nonlinear))[:15]:
        print("  " + repr(h))

    print("\ntop-15 STRONGEST perturbations (by |effect|):")
    for h in sorted(hits, key=lambda h: -abs(h.effect))[:15]:
        print("  " + repr(h))


if __name__ == "__main__":
    _main()
