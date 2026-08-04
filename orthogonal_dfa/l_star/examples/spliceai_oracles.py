r"""Composition-residual and set-difference E-L\* oracles built on
:class:`~orthogonal_dfa.l_star.examples.spliceai_oracle.SpliceModelOracle`.

* :func:`balanced_oracle` / :func:`canonical_oracle` -- the raw SpliceAI (or FM)
  call, with the accept threshold at the median score at a length / at run_model's
  canonical calibration.
* :class:`CompositionResidualOracle` / :class:`PerLengthResidualOracle` -- the exon
  score with its generic bag-of-k-mers composition regressed out (single fit length
  / per-length bins).
* :class:`SetDifferenceOracle` -- ``a \\ b``, to contrast SpliceAI against the FM.

See ``ELSTAR_NEURAL_ORACLE_FINDINGS.md`` for what running E-L\* on these produced.
"""

from typing import List

import numpy as np

from orthogonal_dfa.data.exon import RawExon
from orthogonal_dfa.l_star.examples.spliceai_oracle import (
    SpliceModelOracle,
    calibrated_spliceai_readout,
    flanks,
    median_threshold,
    run_over_middles,
)
from orthogonal_dfa.l_star.structures import Oracle
from orthogonal_dfa.oracle.run_model import calibrate
from orthogonal_dfa.spliceai.exon_score import device_of, spliceai_exon_scores

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


def exon_scores(model, exon, strings, *, chunk=1024):
    """Continuous SpliceAI exon score for each flank-wrapped middle."""
    flank_l, flank_r = flanks(exon)
    return run_over_middles(
        model,
        flank_l,
        flank_r,
        strings,
        spliceai_exon_scores,
        device=device_of(model),
        chunk=chunk,
    )


def balanced_oracle(model, exon, length, *, chunk=1024):
    """Oracle whose accept threshold is the median exon score at middle ``length``."""
    # .function skips the permacache: the FM model is not stable_hashable.
    threshold = median_threshold.function(model, exon, length)
    readout = calibrated_spliceai_readout(threshold, length)
    return SpliceModelOracle(exon, model, readout, chunk=chunk)


def canonical_oracle(model, exon, *, chunk=1024):
    """Oracle using run_model's canonical calibration (threshold at the score mean)."""
    threshold = calibrate(exon, model)["mean"]
    readout = calibrated_spliceai_readout(threshold, exon.random_text_length)
    return SpliceModelOracle(exon, model, readout, chunk=chunk)


def bow_features(strings: List[List[int]], n_max: int) -> np.ndarray:
    """(N, D) generic bag-of-k-mers frequency features for k=1..n_max, D=sum 4^k.

    Position- and frame-agnostic: every k-mer's sliding-window count divided by the
    number of windows. CG and the stop codons are present only implicitly, as
    individual k-mer frequencies among all 4^k -- nothing is hand-embedded.
    """
    N = len(strings)
    D = sum(4**k for k in range(1, n_max + 1))
    F = np.zeros((N, D), dtype=np.float32)
    for i, s in enumerate(strings):
        s = np.asarray(s, dtype=np.int64)
        m = len(s)
        off = 0
        for k in range(1, n_max + 1):
            width = 4**k
            if m >= k:
                ids = np.zeros(m - k + 1, dtype=np.int64)
                for j in range(k):
                    ids = ids * 4 + s[j : m - k + 1 + j]
                counts = np.bincount(ids, minlength=width).astype(np.float32)
                F[i, off : off + width] = counts / (m - k + 1)
            off += width
    return F


class CompositionResidualOracle(Oracle):
    """Exon score with its generic bag-of-k-mers-composition part regressed out.

    Fits a ridge model of the score on n<=n_max k-mer frequencies at a single
    ``ref_len`` and thresholds the residual at its median. Composition removal only
    holds near ``ref_len``; for the full prefix+suffix query-length range use
    :class:`PerLengthResidualOracle`.
    """

    def __init__(
        self,
        exon: RawExon,
        model,
        *,
        n_max=4,
        ref_len=95,
        fit_count=40000,
        ridge=1.0,
        seed=0,
    ):
        self._exon = exon
        self._model = model
        self._n_max = n_max
        self._length = exon.random_text_length
        rng = np.random.default_rng(seed)
        mids = rng.integers(0, 4, size=(fit_count, ref_len)).tolist()
        S = exon_scores(model, exon, mids).astype(np.float64)
        F = bow_features(mids, n_max).astype(np.float64)
        self._Fmean, self._Smean = F.mean(0), S.mean()
        Fc = F - self._Fmean
        A = Fc.T @ Fc + ridge * np.eye(F.shape[1])
        self._beta = np.linalg.solve(A, Fc.T @ (S - self._Smean))
        resid = S - (self._Smean + Fc @ self._beta)
        self._thresh = float(np.median(resid))
        self._r2 = float(1 - (resid**2).sum() / ((S - S.mean()) ** 2).sum())

    @property
    def alphabet_size(self) -> int:
        return 4

    @property
    def string_length(self) -> int:
        return self._length

    @property
    def composition_r2(self) -> float:
        return self._r2

    def residual_scores(self, strings: List[List[int]]) -> np.ndarray:
        raw = exon_scores(self._model, self._exon, strings).astype(np.float64)
        F = bow_features(strings, self._n_max).astype(np.float64)
        return raw - (self._Smean + (F - self._Fmean) @ self._beta)

    def membership_queries(self, strings: List[List[int]], chunk=8192) -> np.ndarray:
        out = np.empty(len(strings), dtype=bool)
        for i in range(0, len(strings), chunk):
            out[i : i + chunk] = (
                self.residual_scores(strings[i : i + chunk]) > self._thresh
            )
        return out

    def membership_query(self, string: List[int]) -> bool:
        return bool(self.membership_queries([string])[0])


class PerLengthResidualOracle(Oracle):
    """Length-robust generic-composition residual.

    Composition's effect on the score is length-dependent, so the length axis is
    binned and a SEPARATE frequency-linear ridge model + median threshold is fit per
    bin. Within a narrow bin length is ~constant, so a frequency model removes
    composition and the per-bin median keeps accept ~= 50%. A query string is scored
    by the model of the bin its length falls in. This keeps the residual balanced and
    (bulk) composition-free across the whole prefix+suffix query-length range; the
    frame-dependent nonlinear stop-codon structure a linear BoW cannot capture is
    intentionally left in the residual.
    """

    def __init__(
        self,
        exon: RawExon,
        model,
        *,
        n_max=4,
        len_lo=95,
        len_hi=190,
        bin_width=5,
        per_bin=8000,
        ridge=1.0,
        seed=0,
    ):
        self._exon = exon
        self._model = model
        self._n_max = n_max
        self._length = exon.random_text_length
        self._edges = np.arange(len_lo, len_hi + bin_width, bin_width)
        rng = np.random.default_rng(seed)
        self._bins = []  # (lo, hi, Fmean, Smean, beta, thresh)
        r2s = []
        for lo, hi in zip(self._edges[:-1], self._edges[1:]):
            lens = rng.integers(lo, hi, size=per_bin)
            mids = [rng.integers(0, 4, size=int(L)).tolist() for L in lens]
            S = exon_scores(model, exon, mids).astype(np.float64)
            F = bow_features(mids, n_max).astype(np.float64)
            Fm, Sm = F.mean(0), S.mean()
            Fc = F - Fm
            A = Fc.T @ Fc + ridge * np.eye(F.shape[1])
            beta = np.linalg.solve(A, Fc.T @ (S - Sm))
            resid = S - (Sm + Fc @ beta)
            self._bins.append((int(lo), int(hi), Fm, Sm, beta, float(np.median(resid))))
            r2s.append(1 - (resid**2).sum() / ((S - S.mean()) ** 2).sum())
        self._r2 = float(np.mean(r2s))

    @property
    def alphabet_size(self) -> int:
        return 4

    @property
    def string_length(self) -> int:
        return self._length

    @property
    def composition_r2(self) -> float:
        return self._r2

    def _bin_for(self, m):
        step = self._edges[1] - self._edges[0]
        i = int(np.clip((m - self._edges[0]) // step, 0, len(self._bins) - 1))
        return self._bins[i]

    def membership_queries(self, strings, chunk=8192):
        out = np.empty(len(strings), dtype=bool)
        by_bin = {}
        for i, s in enumerate(strings):
            by_bin.setdefault(self._bin_for(len(s))[0], []).append(i)
        for _, idxs in by_bin.items():
            for j0 in range(0, len(idxs), chunk):
                subidx = idxs[j0 : j0 + chunk]
                sub = [strings[i] for i in subidx]
                _, _, Fm, Sm, beta, th = self._bin_for(len(sub[0]))
                raw = exon_scores(self._model, self._exon, sub).astype(np.float64)
                F = bow_features(sub, self._n_max).astype(np.float64)
                r = raw - (Sm + (F - Fm) @ beta)
                for k, i in enumerate(subidx):
                    out[i] = r[k] > th
        return out

    def membership_query(self, string):
        return bool(self.membership_queries([string])[0])


class SetDifferenceOracle(Oracle):
    r"""``a \\ b``: membership = ``a`` accepts AND ``b`` does not.

    Used to contrast SpliceAI against the fixed-motif model, e.g. with two balanced
    oracles (plain) or two :class:`CompositionResidualOracle`s (composition stripped
    from both before differencing).
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
