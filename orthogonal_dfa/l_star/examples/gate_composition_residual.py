"""Composition-deconfounded oracle via the repo's monotonic **gate**, run backwards.

:mod:`composition_residual` subtracts a per-length-bin *linear* bag-of-k-mers
prediction from the score.  This module keeps that linear model as the composition
feature phi, then puts a :class:`Monotonic1D` **gate** on top -- the same gate the
residual-training machinery uses -- fit to the score, and subtracts *its* output:

    resid = raw_score - monotonic_bin( intercept_bin + bow_features(middle) @ beta_bin )

That is the composition gate run backwards (``compute_input``: r_prev = r_next -
monotonic(phi)).  Because the monotonic is trained against the *same* score it is
subtracted from, it is already in score units -- no ad-hoc rescaling.  The gate removes
composition strictly more thoroughly than the linear fit (a monotonic reshaping of the
composition index), so its residual is a purer test of non-compositional structure.

Everything is per length bin, exactly like :func:`_fit_composition_bins`, because the
score-to-composition map drifts with length even though the frequency features do not.
"""

import numpy as np
import torch
from permacache import permacache, stable_hash
from torch import nn

from orthogonal_dfa.data.exon import RawExon
from orthogonal_dfa.l_star.examples.composition_residual import (
    _fit_composition_bins,
    bow_features,
)
from orthogonal_dfa.l_star.examples.spliceai_oracle import (
    SpliceModelOracle,
    flanks,
    median_threshold,
    run_over_middles,
)
from orthogonal_dfa.module.monotonic import Monotonic1D
from orthogonal_dfa.spliceai.exon_score import device_of

# Gate hyperparameters, matching train_monotonic_for_manual_dfa's defaults.
MAX_Z_ABS = 4.0
NUM_INPUT_BREAKS = 1000


def _fit_monotonic(pred_lin, scores, device, *, epochs, lr=1e-2, batch=2000, seed=0):
    """Fit a Monotonic1D mapping the linear composition prediction -> score (MSE).

    pred_lin already predicts the score (it is the OLS fit), so the monotonic only
    adds a nonlinear, order-preserving recalibration -- but being trained against the
    score, its output is in score units, so ``raw - monotonic(pred_lin)`` needs no
    ad-hoc coefficient."""
    torch.manual_seed(seed)
    mono = Monotonic1D(MAX_Z_ABS, NUM_INPUT_BREAKS, batch_norm=True).to(device)
    x = torch.as_tensor(pred_lin, dtype=torch.float32, device=device).view(-1, 1)
    y = torch.as_tensor(scores, dtype=torch.float32, device=device).view(-1, 1)
    opt = torch.optim.Adam(mono.parameters(), lr=lr)
    mono.train()
    n = x.shape[0]
    gen = torch.Generator(device=device).manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device, generator=gen)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss = ((mono(x[idx]) - y[idx]) ** 2).mean()
            loss.backward()
            opt.step()
    mono.eval()
    return {k: v.cpu() for k, v in mono.state_dict().items()}


@permacache(
    "orthogonal_dfa/l_star/examples/gate_composition_residual/fit_gate_bins_1",
    key_function=dict(
        score_model=lambda m: stable_hash(m, version=2),
        exon=stable_hash,
        device=lambda _: None,
        chunk=lambda _: None,
    ),
)
def _fit_gate_bins(
    score_model,
    exon: RawExon,
    *,
    n_max=4,
    len_lo=90,
    len_hi=195,
    bin_width=5,
    per_bin=8000,
    epochs=300,
    seed=0,
    device=None,
    chunk=1024,
):
    """Linear composition fit (reused from composition_residual) plus a per-bin
    monotonic gate fit to the score.  Returns the linear fit dict augmented with a
    list of per-bin monotonic state dicts."""
    lin = _fit_composition_bins(
        score_model, exon, n_max=n_max, len_lo=len_lo, len_hi=len_hi,
        bin_width=bin_width, per_bin=per_bin, seed=seed, device=device, chunk=chunk,
    )
    flank_l, flank_r = flanks(exon)
    dev = device_of(score_model, device)
    edges = np.arange(len_lo, len_hi + bin_width, bin_width)
    rng = np.random.default_rng(seed + 1)
    monotonics = []
    for bi, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        lens = rng.integers(lo, hi, size=per_bin)
        # astype, not dtype=np.uint8: the dtype changes the values drawn, and
        # this value is permacached.
        mids = [
            rng.integers(0, 4, size=int(m)).astype(np.uint8).tobytes() for m in lens
        ]
        scores = run_over_middles(
            score_model, flank_l, flank_r, mids, device=dev, chunk=chunk
        ).astype(np.float64)
        feats = bow_features(mids, n_max).astype(np.float64)
        pred_lin = lin["intercepts"][bi] + feats @ lin["betas"][bi]
        monotonics.append(_fit_monotonic(pred_lin, scores, dev, epochs=epochs, seed=seed + bi))
    return dict(**lin, monotonics=monotonics)


class GateCompositionResidualScore(nn.Module):
    """Wraps a score model, subtracting a per-length-bin monotonic-gate prediction of
    the score from the composition index (intercept + bow_features @ beta)."""

    def __init__(
        self, score_model, *, flank_l_len, n_max, edge0, step, intercepts, betas,
        r2s, monotonics,
    ):
        super().__init__()
        self.score_model = score_model
        self._flank_l_len = flank_l_len
        self._n_max = n_max
        self.register_buffer("_edge0", torch.tensor(float(edge0)))
        self.register_buffer("_step", torch.tensor(float(step)))
        self.register_buffer("_intercepts", torch.as_tensor(np.asarray(intercepts, np.float32)))
        self.register_buffer("_betas", torch.as_tensor(np.asarray(betas, np.float32)))
        monos = []
        for sd in monotonics:
            m = Monotonic1D(MAX_Z_ABS, NUM_INPUT_BREAKS, batch_norm=True)
            m.load_state_dict({k: torch.as_tensor(v) for k, v in sd.items()})
            monos.append(m.eval())
        self.monotonics = nn.ModuleList(monos)

    def _bin_indices(self, lengths):
        idx = torch.div(lengths - self._edge0, self._step, rounding_mode="floor").long()
        return idx.clamp(0, self._betas.shape[0] - 1)

    def forward(self, x, lengths):
        raw = self.score_model(x, lengths)
        codes = x.argmax(-1).cpu().numpy()
        lens = lengths.cpu().numpy()
        # #219 made bow_features/symbols read bytes (np.frombuffer), not int lists.
        middles = [
            codes[i, self._flank_l_len : self._flank_l_len + int(m)]
            .astype(np.uint8)
            .tobytes()
            for i, m in enumerate(lens)
        ]
        feats = torch.as_tensor(
            bow_features(middles, self._n_max), device=raw.device, dtype=self._betas.dtype
        )
        idx = self._bin_indices(lengths.to(self._edge0.device))
        pred_lin = self._intercepts[idx] + (feats * self._betas[idx]).sum(-1)  # (N,)
        out = raw.clone()
        for b in torch.unique(idx):
            sel = (idx == b).nonzero(as_tuple=True)[0]
            m = self.monotonics[int(b)](pred_lin[sel].view(-1, 1).to(raw.device)).view(-1)
            out[sel] = raw[sel] - m.to(raw.dtype)
        return out


def fit_gate_composition_residual(score_model, exon, *, device=None, **kw):
    """Fit a GateCompositionResidualScore around score_model (eval mode)."""
    fit = _fit_gate_bins(score_model, exon, device=device, **kw)
    flank_l, _ = flanks(exon)
    module = GateCompositionResidualScore(
        score_model, flank_l_len=len(flank_l), n_max=kw.get("n_max", 4), **{
            k: fit[k] for k in ("edge0", "step", "intercepts", "betas", "r2s", "monotonics")
        }
    )
    return module.to(device_of(score_model, device)).eval()


def gate_residual_oracle(exon, model, *, length, device=None, **kw):
    """The composition-deconfounded (gate-backwards) oracle, median-thresholded."""
    from orthogonal_dfa.spliceai.exon_score import SpliceAIExonScore

    score_model = SpliceAIExonScore(model).eval()
    gscore = fit_gate_composition_residual(score_model, exon, device=device, **kw)
    threshold = median_threshold(gscore, exon, length)
    return SpliceModelOracle(exon, gscore, threshold, device=device)
