r"""SpliceAI (or FM) E-L\* oracle builders, thin wrappers over
:class:`~orthogonal_dfa.l_star.examples.spliceai_oracle.SpliceModelOracle` and
:class:`~orthogonal_dfa.l_star.examples.composition_residual.CompositionResidualScore`.

* :func:`balanced_oracle` / :func:`canonical_oracle` -- the raw call, thresholded at
  the median score at a length / at run_model's canonical calibration.
* :func:`residual_oracle` -- the exon score with its generic bag-of-k-mers composition
  regressed out per length bin (a ``CompositionResidualScore`` module the normal oracle
  wraps), thresholded at the median residual.

To contrast SpliceAI against the FM, wrap two of these in
:class:`~orthogonal_dfa.l_star.examples.set_difference.SetDifferenceOracle` (with the FM
from :func:`~orthogonal_dfa.spliceai.load_model.load_fm`).

See ``ELSTAR_NEURAL_ORACLE_FINDINGS.md`` for what running E-L\* on these produced.
Those runs predate the refactor onto the shared modules; the residual there was a
per-length ridge fit rather than today's drop-column OLS, so a fresh run's exact
numbers and residual DFAs may differ.
"""

from permacache import no_cache_global

from orthogonal_dfa.data.exon import RawExon
from orthogonal_dfa.l_star.examples.composition_residual import fit_composition_residual
from orthogonal_dfa.l_star.examples.spliceai_oracle import (
    SpliceModelOracle,
    median_threshold,
)
from orthogonal_dfa.oracle.run_model import calibrate
from orthogonal_dfa.spliceai.exon_score import SpliceAIExonScore


def score_model_of(model):
    """Wrap a splice model so its forward maps (one-hot x, middle lengths) -> exon score."""
    return SpliceAIExonScore(model).eval()


def balanced_oracle(model, exon: RawExon, length, *, chunk=1024):
    """Oracle thresholding the exon score at its median over random length-``length`` middles."""
    score_model = score_model_of(model)
    # .function skips the permacache: the FM model is not stable_hashable.
    threshold = median_threshold.function(score_model, exon, length)
    return SpliceModelOracle(exon, score_model, threshold, chunk=chunk)


def canonical_oracle(model, exon: RawExon, *, chunk=1024):
    """Oracle using run_model's canonical calibration (threshold at the score mean)."""
    threshold = calibrate(exon, model)["mean"]
    return SpliceModelOracle(exon, score_model_of(model), threshold, chunk=chunk)


def _fit_residual(score_model, exon, **kw):
    try:
        return fit_composition_residual(score_model, exon, **kw)
    except TypeError:
        # A non-stable-hashable model (the FM): recompute without touching the cache.
        with no_cache_global():
            return fit_composition_residual(score_model, exon, **kw)


def residual_oracle(
    model, exon: RawExon, *, n_max=4, len_lo, len_hi, bin_width=5, chunk=1024
):
    """Oracle on the per-length composition residual, thresholded at the median residual.

    The residual is one ``CompositionResidualScore`` module (a bag-of-k-mers fit
    regressed out per length bin); a single median threshold keeps it balanced across
    the band since each bin is centered.  Returns (oracle, mean held-out composition R^2).
    """
    residual = _fit_residual(
        score_model_of(model),
        exon,
        n_max=n_max,
        len_lo=len_lo,
        len_hi=len_hi,
        bin_width=bin_width,
    )
    threshold = median_threshold.function(residual, exon, exon.random_text_length)
    oracle = SpliceModelOracle(exon, residual, threshold, chunk=chunk)
    return oracle, residual.composition_r2
