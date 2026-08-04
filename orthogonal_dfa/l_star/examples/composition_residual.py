"""A score model that regresses generic sequence composition out of the exon score.

The SpliceAI exon score is dominated by low-order sequence composition (CpG-rich up,
AT/stop-rich down).  :class:`CompositionResidualScore` wraps a score model and
subtracts a linear bag-of-k-mers prediction of its score, so what E-L\\* sees is the
composition residual.  Because it acts on the scalar score (not the per-position
logits) and gets the middle from its own input, it is just an ``nn.Module`` the
normal :class:`~orthogonal_dfa.l_star.examples.spliceai_oracle.SpliceModelOracle`
wraps -- no bespoke oracle.  See ``ELSTAR_NEURAL_ORACLE_FINDINGS.md``.
"""

import warnings

import numpy as np
import torch
from permacache import permacache, stable_hash
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
    wrapping oracle balanced across the fitted band.  The fit lives in buffers, so it
    survives ``state_dict``; ``n_max`` and the flank length are architecture, needed
    to reconstruct before loading.  Build with :func:`fit_composition_residual`."""

    def __init__(
        self,
        score_model,
        *,
        flank_l_len,
        n_max,
        edge0,
        step,
        f_means,
        s_means,
        betas,
        r2,
    ):
        super().__init__()
        self.score_model = score_model
        self._flank_l_len = flank_l_len
        self._n_max = n_max
        self.register_buffer("_edge0", torch.tensor(float(edge0)))
        self.register_buffer("_step", torch.tensor(float(step)))
        self.register_buffer(
            "_f_means", torch.as_tensor(np.asarray(f_means, np.float32))
        )
        self.register_buffer(
            "_s_means", torch.as_tensor(np.asarray(s_means, np.float32))
        )
        self.register_buffer("_betas", torch.as_tensor(np.asarray(betas, np.float32)))
        self.register_buffer("_composition_r2", torch.tensor(float(r2)))

    @property
    def composition_r2(self) -> float:
        """Held-out R^2 of the k-mer regression, averaged over bins."""
        return float(self._composition_r2)

    def _bin_indices(self, lengths):
        idx = torch.div(lengths - self._edge0, self._step, rounding_mode="floor").long()
        clamped = idx.clamp(0, self._f_means.shape[0] - 1)
        if bool((idx != clamped).any()):
            hi = float(self._edge0) + self._f_means.shape[0] * float(self._step)
            warnings.warn(
                f"query length outside the fitted band [{float(self._edge0):.0f}, "
                f"{hi:.0f}); the nearest bin is used there and its score is not "
                "calibrated -- widen len_lo/len_hi to cover the queried lengths",
                stacklevel=3,
            )
        return clamped

    def forward(self, x, lengths):
        raw = self.score_model(x, lengths)
        codes = x.argmax(-1).cpu().numpy()
        lens = lengths.cpu().numpy()
        middles = [
            codes[i, self._flank_l_len : self._flank_l_len + int(m)].tolist()
            for i, m in enumerate(lens)
        ]
        feats = torch.as_tensor(
            bow_features(middles, self._n_max),
            device=raw.device,
            dtype=self._betas.dtype,
        )
        idx = self._bin_indices(lengths.to(self._edge0.device))
        pred = self._s_means[idx] + (
            (feats - self._f_means[idx]) * self._betas[idx]
        ).sum(-1)
        return raw - pred.to(raw.dtype)


def _fit_bin(feats, scores, ridge):
    """Standardized, sample-averaged ridge -> (feature_mean, score_mean, coefficients).

    Scale-free by construction: the normal equations are averaged over the samples
    and the predictors are standardized to unit variance, so the penalty is a
    correlation-matrix ridge that does not move with ``per_bin`` or with k-mer order
    (higher-order k-mers are rarer, so on the raw scale their Gram diagonal is
    smaller and a fixed penalty would crush them).  The effective strength is
    ``ridge * D / n`` -- regularization proportional to model complexity per sample,
    lighter with more middles, heavier with more k-mers."""
    n, D = feats.shape
    f_mean = feats.mean(0)
    centered = feats - f_mean
    std = np.sqrt((centered**2).mean(0))
    std[std == 0] = 1.0  # k-mers absent at this length are constant; their coef stays 0
    z = centered / std  # unit-variance predictors -> correlation-matrix Gram below
    coef = np.linalg.solve(
        z.T @ z / n + (ridge * D / n) * np.eye(D), z.T @ (scores - scores.mean()) / n
    )
    return f_mean, scores.mean(), coef / std


def _r2(feats, scores, f_mean, s_mean, beta):
    resid = scores - (s_mean + (feats - f_mean) @ beta)
    return 1 - (resid**2).sum() / ((scores - scores.mean()) ** 2).sum()


@permacache(
    "orthogonal_dfa/l_star/examples/composition_residual/fit_bins",
    key_function=dict(
        score_model=lambda m: stable_hash(m, version=2),
        exon=stable_hash,
        device=lambda _: None,  # does not affect the result
        chunk=lambda _: None,
    ),
)
def _fit_composition_bins(
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
    """Ridge-fit the k-mer regression per length bin; returns the picklable fit (small
    numpy arrays), so the cache does not have to pickle the model.

    The deployed coefficients use every middle; the reported r2 is held out (a 20%
    slice), since in-sample r2 with up to 340 features reads optimistically."""
    flank_l, flank_r = flanks(exon)
    dev = device_of(score_model, device)
    edges = np.arange(len_lo, len_hi + bin_width, bin_width)
    assert len(edges) >= 2, (
        f"need at least one length bin, got none: len_hi ({len_hi}) must exceed "
        f"len_lo ({len_lo}) -- len_hi is exclusive, so a single length L is len_hi=L+1"
    )
    rng = np.random.default_rng(seed)
    f_means, s_means, betas, r2s = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        lens = rng.integers(lo, hi, size=per_bin)
        mids = [rng.integers(0, 4, size=int(length)).tolist() for length in lens]
        scores = run_over_middles(
            score_model, flank_l, flank_r, mids, device=dev, chunk=chunk
        ).astype(np.float64)
        feats = bow_features(mids, n_max).astype(np.float64)
        f_mean, s_mean, beta = _fit_bin(feats, scores, ridge)
        cut = max(1, per_bin - max(1, per_bin // 5))
        held = _fit_bin(feats[:cut], scores[:cut], ridge)
        f_means.append(f_mean)
        s_means.append(s_mean)
        betas.append(beta)
        r2s.append(_r2(feats[cut:], scores[cut:], *held))
    return dict(
        edge0=float(edges[0]),
        step=float(edges[1] - edges[0]),
        f_means=np.stack(f_means).astype(np.float32),
        s_means=np.array(s_means, dtype=np.float32),
        betas=np.stack(betas).astype(np.float32),
        r2=float(np.mean(r2s)),
    )


def _residual_module(score_model, exon, n_max, fit, device=None):
    flank_l, _ = flanks(exon)
    module = CompositionResidualScore(
        score_model, flank_l_len=len(flank_l), n_max=n_max, **fit
    )
    return module.to(device_of(score_model, device)).eval()


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
    cache=True,
):
    """Fit a :class:`CompositionResidualScore` around ``score_model`` over the length
    band ``[len_lo, len_hi)`` (``len_hi`` exclusive, as in ``rng.integers``; a single
    length L is ``len_hi = L+1``).

    The band must cover the exon's query length; E-L* also queries other lengths
    (prefix+suffix), which the module warns about rather than silently miscalibrating.

    The per-bin fit is permacached on ``stable_hash(score_model)``.  Like
    ``median_threshold``, that hash only works for a stable-hashable model; SpliceAI
    and the FM are not (a bare function / a weakref), so pass ``cache=False`` for
    them to recompute rather than raise on the cache key."""
    q = exon.random_text_length
    assert len_lo <= q < len_hi, (
        f"the exon's query length {q} is outside the fitted band [{len_lo}, {len_hi}); "
        "widen len_lo/len_hi so the band covers the lengths E-L* will query"
    )
    fit_bins = _fit_composition_bins if cache else _fit_composition_bins.function
    fit = fit_bins(
        score_model,
        exon,
        n_max=n_max,
        len_lo=len_lo,
        len_hi=len_hi,
        bin_width=bin_width,
        per_bin=per_bin,
        ridge=ridge,
        seed=seed,
        device=device,
        chunk=chunk,
    )
    return _residual_module(score_model, exon, n_max, fit, device)
