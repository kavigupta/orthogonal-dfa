"""Like :mod:`composition_residual`, but subtracts a monotonic function of the linear
composition prediction rather than the linear prediction itself:

    resid = raw_score - monotonic_bin( intercept_bin + bow_features(middle) @ beta_bin )

The monotonic (a :class:`Monotonic1D`) is fit per length bin by MSE against the score,
so its output is already in score units.  It can only reshape the composition index
monotonically, so it removes at least as much composition as the linear fit.
"""

import numpy as np
import torch
from permacache import permacache, stable_hash
from torch import nn

from orthogonal_dfa.data.exon import RawExon
from orthogonal_dfa.l_star.examples.composition_residual import (
    CompositionResidualScore,
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

# Monotonic hyperparameters, matching train_monotonic_for_manual_dfa's defaults.
MAX_Z_ABS = 4.0
NUM_INPUT_BREAKS = 1000


def _fit_monotonic(pred_lin, scores, device, *, epochs, lr=1e-2, batch=2000, seed=0):
    """Fit a Monotonic1D mapping the linear composition prediction -> score (MSE).

    Returns the fitted state dict."""
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
    monotonic fit to the score.  Returns the linear fit dict augmented with a
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
        mids = [rng.integers(0, 4, size=int(m)).tolist() for m in lens]
        scores = run_over_middles(
            score_model, flank_l, flank_r, mids, device=dev, chunk=chunk
        ).astype(np.float64)
        feats = bow_features(mids, n_max).astype(np.float64)
        pred_lin = lin["intercepts"][bi] + feats @ lin["betas"][bi]
        monotonics.append(_fit_monotonic(pred_lin, scores, dev, epochs=epochs, seed=seed + bi))
    return dict(**lin, monotonics=monotonics)


class GateCompositionResidualScore(CompositionResidualScore):
    """CompositionResidualScore that subtracts ``monotonic_bin(pred_lin)`` instead of
    the bare composition index ``pred_lin``."""

    def __init__(self, score_model, *, monotonics, **kw):
        super().__init__(score_model, **kw)
        monos = []
        for sd in monotonics:
            m = Monotonic1D(MAX_Z_ABS, NUM_INPUT_BREAKS, batch_norm=True)
            m.load_state_dict({k: torch.as_tensor(v) for k, v in sd.items()})
            monos.append(m.eval())
        self.monotonics = nn.ModuleList(monos)

    def forward(self, x, lengths):
        raw = self.score_model(x, lengths)
        pred_lin, idx = self._composition_pred(x, lengths)
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
