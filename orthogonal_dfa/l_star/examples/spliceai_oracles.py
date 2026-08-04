r"""Composition-residual and set-difference E-L\* oracles, thin wrappers over
:class:`~orthogonal_dfa.l_star.examples.spliceai_oracle.SpliceModelOracle` and
:class:`~orthogonal_dfa.l_star.examples.composition_residual.CompositionResidualScore`.

* :func:`balanced_oracle` / :func:`canonical_oracle` -- the raw SpliceAI (or FM)
  call, thresholded at the median score at a length / at run_model's canonical
  calibration.
* :func:`residual_oracle` -- the exon score with its generic bag-of-k-mers
  composition regressed out per length bin (a ``CompositionResidualScore`` module the
  normal oracle wraps), thresholded at the median residual.
* :class:`SetDifferenceOracle` -- ``a \\ b``, to contrast SpliceAI against the FM.

See ``ELSTAR_NEURAL_ORACLE_FINDINGS.md`` for what running E-L\* on these produced.
Those runs predate the refactor onto the shared modules; the residual there was a
per-length ridge fit rather than today's drop-column OLS, so a fresh run's exact
numbers and residual DFAs may differ.
"""

from typing import List

import numpy as np
from permacache import no_cache_global

from orthogonal_dfa.data.exon import RawExon
from orthogonal_dfa.l_star.examples.composition_residual import fit_composition_residual
from orthogonal_dfa.l_star.examples.spliceai_oracle import (
    SpliceModelOracle,
    median_threshold,
)
from orthogonal_dfa.l_star.structures import Oracle
from orthogonal_dfa.oracle.run_model import calibrate
from orthogonal_dfa.spliceai.exon_score import SpliceAIExonScore

# The fixed-motif ("FM") model is a trained modular_splicing model living in a
# separate repo on this machine (BothLSSIModels + an 82-motif RBNS PSAMMotifModel).
FM_REPO = "/mnt/md0/ExpeditionsCommon/spliceai/Canonical"
FM_MODEL_PREFIX = f"{FM_REPO}/model/msp-273.665a3"


def load_fm(seed=1):
    """Load the fixed-motif model (seed 1..5) as an eval/cuda nn.Module."""
    import sys

    if FM_REPO not in sys.path:
        sys.path.insert(0, FM_REPO)
    # modular_splicing is an external repo added to sys.path above, not a dependency.
    from modular_splicing.utils.io import load_model  # pylint: disable=import-error

    _, model = load_model(f"{FM_MODEL_PREFIX}_{seed}")  # picks the latest step
    return model.eval().cuda()


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


class SetDifferenceOracle(Oracle):
    r"""``a \\ b``: membership = ``a`` accepts AND ``b`` does not.

    Used to contrast SpliceAI against the fixed-motif model, e.g. with two balanced
    oracles (plain) or two residual oracles (composition stripped from both before
    differencing).
    """

    def __init__(self, oracle_a, oracle_b, exon: RawExon):
        self._a = oracle_a
        self._b = oracle_b
        self._length = exon.random_text_length

    @property
    def alphabet_size(self) -> int:
        return 4

    @property
    def string_length(self) -> int:
        return self._length

    def membership_queries(self, strings: List[List[int]]) -> np.ndarray:
        return self._a.membership_queries(strings) & ~self._b.membership_queries(
            strings
        )

    def membership_query(self, string: List[int]) -> bool:
        return bool(self.membership_queries([string])[0])
