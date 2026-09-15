import warnings

import numpy as np
import torch
from permacache import permacache, stable_hash
from torch import nn

from orthogonal_dfa.data.exon import RawExon
from orthogonal_dfa.l_star.examples.spliceai_oracle import (
    flanks,
    run_over_middles,
    symbols,
)
from orthogonal_dfa.spliceai.exon_score import device_of

# Minimal number of samples per free coefficient to fit an unregularized linear model effectively.
MIN_SAMPLES_PER_PARAMETER = 20

# Singular values below this fraction of the largest are dropped by :func:`_fit_bin`.
# The k and k-1 blocks of bow_features are near-dependent -- summing a k-block over
# its last base recovers the (k-1)-mer frequency up to an O(1/m) boundary term -- so
# the design is full rank but ill-conditioned (~3e6 at n_max=4).  Keeping those
# directions lets the fit spend enormous coefficients on boundary effects.
RCOND = 1e-4


def free_parameters(k_max):
    """Width of :func:`bow_features`, hence the coefficient count a fit estimates."""
    return sum(4**k - 1 for k in range(1, k_max + 1))


def bow_features(strings, k_max):
    """
    (N, D) bag-of-k-mers design matrix for k=1..k_max, D=sum(4**k - 1).

    Each k's block holds the sliding-window frequency of every k-mer but the last,
    to avoid linear dependence (since they sum to 1 in each block).
    """
    D = free_parameters(k_max)
    F = np.zeros((len(strings), D), dtype=np.float32)
    for i, s in enumerate(strings):
        s = symbols(s)
        m, off = len(s), 0
        for k in range(1, k_max + 1):
            width = 4**k - 1
            if m >= k:
                ids = np.zeros(m - k + 1, dtype=np.int64)
                for j in range(k):
                    ids = ids * 4 + s[j : m - k + 1 + j]
                counts = np.bincount(ids, minlength=width + 1)[:width]
                F[i, off : off + width] = counts / (m - k + 1)
            off += width
    return F


class CompositionResidualScore(nn.Module):
    """
    Wraps a score model, subtracting a per-length-bin linear bag-of-k-mers
    prediction of its score.

    Each bin's linear model carries an intercept, so the residual is centered per
    bin.

    The fit is one linear model per length bin, written as parallel arrays.

    With B bins and D = sum(4**k - 1 for k in 1..n_max) bow_features, a middle of
    length m scores

        bin  = clamp(floor((m - edge0) / step), 0, B - 1)
        pred = intercepts[bin] + bow_features(middle) @ betas[bin]

    where
        - edge0, step: the band start and bin width (scalars), i.e. len_lo and
          bin_width, from which every bin edge follows.
        - intercepts: (B,) each bin's constant term.
        - betas: (B, D) each bin's k-mer coefficients.
        - r2s: (B,) diagnostic only, surfaced per bin as composition_r2s and
          averaged as composition_r2.

    score_model, flank_l_len and n_max are architecture rather than fit: they are
    needed to reconstruct the module before load_state_dict can restore the rest.
    """

    def __init__(
        self,
        score_model,
        *,
        flank_l_len,
        n_max,
        edge0,
        step,
        intercepts,
        betas,
        r2s,
    ):
        super().__init__()
        self.score_model = score_model
        self._flank_l_len = flank_l_len
        self._n_max = n_max
        self.register_buffer("_edge0", torch.tensor(float(edge0)))
        self.register_buffer("_step", torch.tensor(float(step)))
        self.register_buffer(
            "_intercepts", torch.as_tensor(np.asarray(intercepts, np.float32))
        )
        self.register_buffer("_betas", torch.as_tensor(np.asarray(betas, np.float32)))
        self.register_buffer(
            "_composition_r2s", torch.as_tensor(np.asarray(r2s, np.float32))
        )

    @property
    def composition_r2s(self) -> np.ndarray:
        """Held-out R^2 of the k-mer regression, per length bin."""
        return self._composition_r2s.cpu().numpy()

    @property
    def composition_r2(self) -> float:
        """Held-out R^2 of the k-mer regression, averaged over bins."""
        return float(self._composition_r2s.mean())

    def _bin_indices(self, lengths):
        idx = torch.div(lengths - self._edge0, self._step, rounding_mode="floor").long()
        clamped = idx.clamp(0, self._betas.shape[0] - 1)
        if bool((idx != clamped).any()):
            hi = float(self._edge0) + self._betas.shape[0] * float(self._step)
            warnings.warn(
                f"query length outside the fitted band [{float(self._edge0):.0f}, "
                f"{hi:.0f}); the nearest bin is used there and its score is not "
                "calibrated -- widen len_lo/len_hi to cover the queried lengths",
                stacklevel=3,
            )
        return clamped

    def _composition_pred(self, x, lengths):
        """Per-row composition index (intercept + bow_features @ beta) and the bin
        index each row falls in."""
        codes = x.argmax(-1).cpu().numpy()
        lens = lengths.cpu().numpy()
        middles = [
            codes[i, self._flank_l_len : self._flank_l_len + int(m)]
            .astype(np.uint8)
            .tobytes()
            for i, m in enumerate(lens)
        ]
        feats = torch.as_tensor(
            bow_features(middles, self._n_max),
            device=self._betas.device,
            dtype=self._betas.dtype,
        )
        idx = self._bin_indices(lengths.to(self._edge0.device))
        pred = self._intercepts[idx] + (feats * self._betas[idx]).sum(-1)
        return pred, idx

    def forward(self, x, lengths):
        raw = self.score_model(x, lengths)
        pred, _ = self._composition_pred(x, lengths)
        return raw - pred.to(raw.device).to(raw.dtype)


def _fit_bin(feats, scores):
    """
    Least squares -> (intercept, coefficients), predicting ``intercept + feats @ coef``.

    Unpenalized (see MIN_SAMPLES_PER_PARAMETER), but truncated at RCOND rather than
    at machine precision, which keeps the coefficients on the scale of the score
    instead of the 1e5 the ill-conditioned directions would otherwise reach.

    Fitting centers the features, but the centering point folds into the intercept
    rather than being kept.
    """
    f_mean, s_mean = feats.mean(0), scores.mean()
    coef, *_ = np.linalg.lstsq(feats - f_mean, scores - s_mean, rcond=RCOND)
    return s_mean - f_mean @ coef, coef


def _r2(feats, scores, intercept, beta):
    resid = scores - (intercept + feats @ beta)
    return 1 - (resid**2).sum() / ((scores - scores.mean()) ** 2).sum()


@permacache(
    # Bump this whenever bow_features' layout or _fit_bin's math changes: the key
    # covers only the arguments, so a stale entry is a silently wrong residual.
    "orthogonal_dfa/l_star/examples/composition_residual/fit_bins_2",
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
    seed=0,
    device=None,
    chunk=1024,
):
    """
    OLS-fit the k-mer regression per length bin; returns the picklable fit (small
    numpy arrays), so the cache does not have to pickle the model.

    We use the whole dataset to fit the linear model, and separately fit a model
    to 80% of the data and score it on the remaining 20% to get a held-out R^2 per bin.
    This is done to be maximally sample efficient in the oracle, which is much more costly
    to run than the linear regression.
    """
    flank_l, flank_r = flanks(exon)
    dev = device_of(score_model, device)
    edges = np.arange(len_lo, len_hi + bin_width, bin_width)
    assert len(edges) >= 2, (
        f"need at least one length bin, got none: len_hi ({len_hi}) must exceed "
        f"len_lo ({len_lo}) -- len_hi is exclusive, so a single length L is len_hi=L+1"
    )
    free = free_parameters(n_max)
    assert per_bin >= MIN_SAMPLES_PER_PARAMETER * free, (
        f"per_bin={per_bin} is too small for n_max={n_max}: the fit is unregularized "
        f"and estimates {free} free coefficients per bin, so it needs at least "
        f"{MIN_SAMPLES_PER_PARAMETER * free} middles to stay out of the overfitting "
        f"regime; raise per_bin or lower n_max"
    )
    rng = np.random.default_rng(seed)
    intercepts, betas, r2s = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        lens = rng.integers(lo, hi, size=per_bin)
        # astype, not dtype=np.uint8: the dtype changes the values drawn, and
        # this value is permacached.
        mids = [
            rng.integers(0, 4, size=int(length)).astype(np.uint8).tobytes()
            for length in lens
        ]
        scores = run_over_middles(
            score_model, flank_l, flank_r, mids, device=dev, chunk=chunk
        ).astype(np.float64)
        feats = bow_features(mids, n_max).astype(np.float64)
        intercept, beta = _fit_bin(feats, scores)
        intercepts.append(intercept)
        betas.append(beta)
        # Scoring the fit above on its own training data would read optimistically, so
        # r2 comes from a throwaway second fit on 80% scored on the 20% it never saw.
        # The middles are i.i.d., so a contiguous split is a random one.
        cut = per_bin * 4 // 5
        probe = _fit_bin(feats[:cut], scores[:cut])
        r2s.append(_r2(feats[cut:], scores[cut:], *probe))
    return dict(
        edge0=float(edges[0]),
        step=float(edges[1] - edges[0]),
        intercepts=np.array(intercepts, dtype=np.float32),
        betas=np.stack(betas).astype(np.float32),
        r2s=np.array(r2s),
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
    seed=0,
    device=None,
    chunk=1024,
):
    """
    Fit a CompositionResidualScore around score_model over the length
    band [len_lo, len_hi).

    The band must cover the exon's query length; E-L* also queries other lengths
    (prefix+suffix), which the module warns about rather than silently miscalibrating.
    """
    q = exon.random_text_length
    assert len_lo <= q < len_hi, (
        f"the exon's query length {q} is outside the fitted band [{len_lo}, {len_hi}); "
        "widen len_lo/len_hi so the band covers the lengths E-L* will query"
    )
    fit = _fit_composition_bins(
        score_model,
        exon,
        n_max=n_max,
        len_lo=len_lo,
        len_hi=len_hi,
        bin_width=bin_width,
        per_bin=per_bin,
        seed=seed,
        device=device,
        chunk=chunk,
    )
    return _residual_module(score_model, exon, n_max, fit, device)
