"""A score model that regresses generic sequence composition out of the exon score.

The SpliceAI exon score is dominated by low-order sequence composition (CpG-rich up,
AT/stop-rich down).  :class:`CompositionResidualScore` wraps a score model and
subtracts a linear bag-of-k-mers prediction of its score, so what E-L\\* sees is the
composition residual.  Because it acts on the scalar score (not the per-position
logits) and gets the middle from its own input, it is just an ``nn.Module`` the
normal :class:`~orthogonal_dfa.l_star.examples.spliceai_oracle.SpliceModelOracle`
wraps -- no bespoke oracle.  See ``ELSTAR_NEURAL_ORACLE_FINDINGS.md``.
"""

import numpy as np
import torch
from torch import nn

from orthogonal_dfa.data.exon import RawExon
from orthogonal_dfa.l_star.examples.spliceai_oracle import flanks, run_over_middles
from orthogonal_dfa.spliceai.exon_score import device_of


def bow_features(strings, n_max):
    """(N, D) generic bag-of-k-mers frequency features for k=1..n_max, D=sum 4^k.

    Position- and frame-agnostic: every k-mer's sliding-window count over the number
    of windows.  CG and the stop codons are present only as individual k-mer
    frequencies among all 4^k -- nothing is hand-embedded."""
    D = sum(4**k for k in range(1, n_max + 1))
    F = np.zeros((len(strings), D), dtype=np.float32)
    for i, s in enumerate(strings):
        s = np.asarray(s, dtype=np.int64)
        m, off = len(s), 0
        for k in range(1, n_max + 1):
            width = 4**k
            if m >= k:
                ids = np.zeros(m - k + 1, dtype=np.int64)
                for j in range(k):
                    ids = ids * 4 + s[j : m - k + 1 + j]
                F[i, off : off + width] = np.bincount(ids, minlength=width) / (
                    m - k + 1
                )
            off += width
    return F


class CompositionResidualScore(nn.Module):
    """Wraps a score model, subtracting a per-length-bin linear bag-of-k-mers
    prediction of its score.

    Each bin's linear model carries an intercept, so the residual is centered per
    bin: its median is ~0 at every length, and one ``median_threshold`` keeps the
    wrapping oracle balanced across the query-length range.  Build with
    :func:`fit_composition_residual`."""

    def __init__(self, score_model, *, flank_l_len, n_max, edges, bins, r2):
        super().__init__()
        self.score_model = score_model
        self.composition_r2 = r2
        self._flank_l_len = flank_l_len
        self._n_max = n_max
        self._edges = edges
        self._bins = bins  # list of (feature_mean, score_mean, coefficients)

    def _bin(self, m):
        step = self._edges[1] - self._edges[0]
        idx = int(np.clip((m - self._edges[0]) // step, 0, len(self._bins) - 1))
        return self._bins[idx]

    def forward(self, x, lengths):
        raw = self.score_model(x, lengths)
        codes = x.argmax(-1).cpu().numpy()
        lens = lengths.cpu().numpy()
        middles = [
            codes[i, self._flank_l_len : self._flank_l_len + int(m)].tolist()
            for i, m in enumerate(lens)
        ]
        feats = bow_features(middles, self._n_max).astype(np.float64)
        pred = np.empty(len(lens))
        for i, m in enumerate(lens):
            f_mean, s_mean, beta = self._bin(int(m))
            pred[i] = s_mean + (feats[i] - f_mean) @ beta
        return raw - torch.as_tensor(pred, device=raw.device, dtype=raw.dtype)


def _fit_bin(feats, scores, ridge):
    """Ridge-regress ``scores`` on ``feats``, returning (feature_mean, score_mean,
    coefficients, r2).

    Scale-free by construction: the normal equations are averaged over the samples
    and the predictors are standardized to unit variance, so the penalty is a
    correlation-matrix ridge that does not move with ``per_bin`` or with k-mer order
    (higher-order k-mers are rarer, so on the raw scale their Gram diagonal is
    smaller and a fixed penalty would crush them).  The effective strength is
    ``ridge * D / n`` -- regularization proportional to model complexity per sample,
    lighter with more middles, heavier with more k-mers."""
    n, D = feats.shape
    f_mean, s_mean = feats.mean(0), scores.mean()
    centered, target = feats - f_mean, scores - s_mean
    std = np.sqrt((centered**2).mean(0))
    std[std == 0] = 1.0  # k-mers absent at this length are constant; their coef stays 0
    z = centered / std  # unit-variance predictors -> correlation-matrix Gram below
    coef = np.linalg.solve(z.T @ z / n + (ridge * D / n) * np.eye(D), z.T @ target / n)
    beta = coef / std
    resid = target - centered @ beta
    r2 = 1 - (resid**2).sum() / (target**2).sum()
    return f_mean, s_mean, beta, r2


def fit_composition_residual(
    score_model,
    exon: RawExon,
    *,
    n_max=4,
    len_lo=95,
    len_hi=190,
    bin_width=5,
    per_bin=8000,
    ridge=1.0,
    seed=0,
    device=None,
    chunk=1024,
):
    """Fit a :class:`CompositionResidualScore` around ``score_model``.

    Per length bin, ridge-regress the exon score on n<=n_max k-mer frequencies over
    random middles and keep the residual (see :func:`_fit_bin` for the scale-free
    ridge).  ``len_hi`` is exclusive (as in ``rng.integers``), so
    ``len_hi == len_lo + 1`` with ``bin_width=1`` is the single-length fit."""
    flank_l, flank_r = flanks(exon)
    dev = device_of(score_model, device)
    edges = np.arange(len_lo, len_hi + bin_width, bin_width)
    assert len(edges) >= 2, (
        f"need at least one length bin, got none: len_hi ({len_hi}) must exceed "
        f"len_lo ({len_lo}) -- len_hi is exclusive, so a single length L is len_hi=L+1"
    )
    rng = np.random.default_rng(seed)
    bins, r2s = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        lens = rng.integers(lo, hi, size=per_bin)
        mids = [rng.integers(0, 4, size=int(length)).tolist() for length in lens]
        scores = run_over_middles(
            score_model, flank_l, flank_r, mids, device=dev, chunk=chunk
        ).astype(np.float64)
        feats = bow_features(mids, n_max).astype(np.float64)
        f_mean, s_mean, beta, r2 = _fit_bin(feats, scores, ridge)
        bins.append((f_mean, s_mean, beta))
        r2s.append(r2)
    return CompositionResidualScore(
        score_model,
        flank_l_len=len(flank_l),
        n_max=n_max,
        edges=edges,
        bins=bins,
        r2=float(np.mean(r2s)),
    ).eval()
