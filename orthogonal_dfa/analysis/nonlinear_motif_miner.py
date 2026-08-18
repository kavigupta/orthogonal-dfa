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
    use_gate: bool = False,
    **fit_kw,
) -> Callable[[Sequence[Sequence[int]]], np.ndarray]:
    """A function ``middles -> controlled scores`` for the residual oracle.

    ``middles`` is a list of equal-length int sequences (A=0,C=1,G=2,T=3).  With
    ``use_gate=False`` (default) composition is removed by the *linear* bag-of-k-mers
    fit (``fit_composition_residual``); with ``use_gate=True`` by the *monotonic gate*
    (``fit_gate_composition_residual``), which also removes monotone-nonlinear
    composition, so any motif that survives it is not just leftover composition.  The
    fit is permacached; ``fit_kw`` (``len_lo``, ``len_hi``, ``n_max`` ...) forwards on.
    """
    score_model = SpliceAIExonScore(spliceai_model).eval()
    if use_gate:
        # optional: the monotonic-gate oracle (PR #194); only needed for the linear-vs-
        # gate comparison, so imported lazily and tolerated if it is not on this branch.
        from orthogonal_dfa.l_star.examples.gate_composition_residual import (  # pylint: disable=import-error,import-outside-toplevel
            fit_gate_composition_residual,
        )

        residual = fit_gate_composition_residual(score_model, exon, device=device, **fit_kw)
    else:
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


# --- in-context motif significance (position-agnostic) ----------------------------


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
    effect: float      # SIGNED mean, over contexts, of the motif's effect relative to
    #                    the average k-mer at the same (background, position)
    tstat: float       # effect / SE across contexts -- directional significance
    magnitude: float   # mean |relative effect| -- how big an effect it makes REGARDLESS
    #                    of sign (captures context-dependent, sign-flipping motifs that
    #                    the signed mean cancels out)
    mag_z: float       # (magnitude - mean over motifs) / std over motifs -- how much
    #                    LARGER an effect than the alternative k-mers, in magnitude
    n: int

    def __repr__(self):
        return (
            f"{self.motif}  |effect|={self.magnitude:.4f} (mag_z={self.mag_z:+5.1f})  "
            f"signed={self.effect:+.4f} (t={self.tstat:+6.1f})  n={self.n}"
        )


def context_motif_significance(
    score: Callable,
    contexts: List[Tuple[List[int], int]],
    motif_k: int,
    *,
    chunk_contexts: int = 200,
) -> List[ContextMotifStat]:
    """For every k-mer, its **in-context** effect measured *against the alternatives*.

    At each context we overwrite ``[p, p+k)`` with every k-mer and record the score
    change ``delta``.  The removed background bases and the position are shared by all
    k-mers at that context, so subtracting the per-context mean over k-mers,
    ``rel = delta - mean_kmers(delta)``, isolates the *identity* of the inserted motif --
    how much more (or less) it moves the score than a typical substitution at that exact
    spot.

    Two summaries per motif, because direction matters:

    - ``effect`` / ``tstat``: the SIGNED mean of ``rel`` across contexts.  Flags motifs
      with a consistent-direction effect; a motif whose sign flips by context cancels.
    - ``magnitude`` / ``mag_z``: ``mean|rel|``, the size of the effect regardless of
      sign, and how many standard deviations that exceeds the average k-mer's.  This is
      the one to rank by for "creates an effect larger than alternatives", because it
      does not cancel a context-dependent (sign-flipping) motif.
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

    mean_rel = rel.mean(0)
    se = rel.std(0, ddof=1) / np.sqrt(rel.shape[0])
    tstat = np.divide(mean_rel, se, out=np.zeros_like(mean_rel), where=se > 0)

    magnitude = np.abs(rel).mean(0)  # direction-agnostic effect size per motif
    mag_z = (magnitude - magnitude.mean()) / magnitude.std()

    return [
        ContextMotifStat(
            motif="".join(BASES[c] for c in m),
            effect=float(mean_rel[i]),
            tstat=float(tstat[i]),
            magnitude=float(magnitude[i]),
            mag_z=float(mag_z[i]),
            n=rel.shape[0],
        )
        for i, m in enumerate(motifs)
    ]


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
    """Harvest in-context motifs from the linear-controlled SpliceAI-400 oracle:
    perturbations whose effect, over consistent backgrounds, is statistically larger
    than alternative substitutions at the same spot -- position-agnostic."""
    import os

    from orthogonal_dfa.data.exon import default_exon
    from orthogonal_dfa.spliceai.load_model import load_spliceai

    n_ctx = int(os.environ.get("N_CTX", "3000"))
    motif_k = int(os.environ.get("MOTIF_K", "3"))
    use_gate = os.environ.get("USE_GATE", "") not in ("", "0", "false")
    length = default_exon.random_text_length
    # optionally restrict positions to the interior, away from the position-locked
    # donor/acceptor edges, so what surfaces is genuinely context- not edge-driven
    margin = int(os.environ.get("EDGE_MARGIN", "0"))
    pos_range = (margin, length - motif_k + 1 - margin) if margin else None

    kind = "GATE" if use_gate else "linear"
    print(f"building {kind}-controlled score (SpliceAI-400)  exon length={length}",
          flush=True)
    score = build_controlled_score(default_exon, load_spliceai(400, 0), use_gate=use_gate)

    print(f"in-context harvest: {n_ctx} contexts, {motif_k}-mers, "
          f"pos_range={pos_range or 'all'}", flush=True)
    contexts = sample_contexts(length, motif_k, n_ctx, pos_range=pos_range)
    stats = context_motif_significance(score, contexts, motif_k)

    print("\nmotifs that create the LARGEST in-context effect vs alternatives "
          "(by |effect|, direction-agnostic):")
    for s in sorted(stats, key=lambda s: -s.magnitude)[:15]:
        print("  " + repr(s))

    print("\n(for reference) most consistent-DIRECTION effects, by signed t:")
    for s in sorted(stats, key=lambda s: -abs(s.tstat))[:8]:
        print("  " + repr(s))


if __name__ == "__main__":
    _main()
