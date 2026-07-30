"""E-L\* membership oracles backed by the SpliceAI neural model (and variants).

A membership query is a variable-length string over {A,C,G,T} (ints 0-3) that fills
the variable middle region of an ``exon``; the fixed flanks are prepended/appended
exactly as :func:`orthogonal_dfa.data.sample_text.sample_text` does, and the model's
per-position (null/acceptor/donor) logits are read at the two exon boundaries and
averaged into a scalar exon score (mirroring
:func:`orthogonal_dfa.oracle.run_model.compute_exon_scores`). Membership is that
score thresholded to a hard accept/reject call.

Oracles provided:

* :class:`SpliceModelOracle` -- the raw model call. Works for any model whose
  ``forward`` produces ``(N, L-cl, 3)`` logits (SpliceAI via ``load_spliceai`` or
  the fixed-motif ``modular_splicing`` model via :func:`load_fm`).
* :class:`CompositionResidualOracle` -- the score with its
  bag-of-k-mers-composition-predictable part regressed out (single fit length).
* :class:`PerLengthResidualOracle` -- the length-robust version of the above; the
  composition model is refit per length bin so the residual stays balanced and
  composition-free across the whole prefix+suffix query-length range.
* :class:`SetDifferenceOracle` -- ``a \\ b`` (a accepts and b does not), used to
  contrast SpliceAI against the fixed-motif model.

See ``ELSTAR_NEURAL_ORACLE_FINDINGS.md`` for what running E-L\* on these produced.
"""

from typing import List

import numpy as np
import torch

from orthogonal_dfa.data.exon import RawExon
from orthogonal_dfa.l_star.structures import Oracle
from orthogonal_dfa.oracle.run_model import batched_run, calibrate

# The fixed-motif ("FM") model is a trained modular_splicing model living in a
# separate repo on this machine (BothLSSIModels + an 82-motif RBNS PSAMMotifModel).
# It loads through modular_splicing's renamed-symbol unpickler; see load_fm.
FM_REPO = "/mnt/md0/ExpeditionsCommon/spliceai/Canonical"
FM_MODEL_PREFIX = f"{FM_REPO}/model/msp-273.665a3"


def load_fm(seed=1):
    """Load the fixed-motif model (seed 1..5) as an eval/cuda nn.Module."""
    import sys

    if FM_REPO not in sys.path:
        sys.path.insert(0, FM_REPO)
    from modular_splicing.utils.io import load_model  # noqa: E402

    _, model = load_model(f"{FM_MODEL_PREFIX}_{seed}")  # picks the latest step
    return model.eval().cuda()


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


class SpliceModelOracle(Oracle):
    """The model's hard exon call on the variable middle as an E-L\* oracle.

    ``calib_len`` re-centers the accept threshold on the median score at that middle
    length (so accept ~= 50% there). This is needed because the exon score drifts
    with length, and it also side-steps the permacached ``calibrate`` (whose
    stable_hash cannot serialize every model). Pass ``calib_len=None`` to use the
    canonical L=189 calibration (SpliceAI only).
    """

    def __init__(self, exon: RawExon, model, calib_len=None):
        self._exon = exon
        self._model = model
        trim = exon.cl // 2 + 2
        self._flank_l = np.array(exon.text[:trim], dtype=np.int64)
        self._flank_r = np.array(exon.text[-trim:], dtype=np.int64)
        self._length = exon.random_text_length
        if calib_len is not None:
            rng = np.random.default_rng(0xCA11B)
            mids = rng.integers(0, 4, size=(20000, calib_len)).tolist()
            self._mean = float(np.median(self._raw_scores(mids)))
            self._std = 1.0  # sign is all that matters for the hard label
        else:
            cal = calibrate(exon, model)
            self._mean = cal["mean"]
            self._std = cal["std"]

    @property
    def alphabet_size(self) -> int:
        return 4

    @property
    def string_length(self) -> int:
        return self._length

    def _raw_scores(self, strings: List[List[int]]) -> np.ndarray:
        """Continuous (acceptor+donor)/2 log-prob exon score for each string.

        E-L\* passes ragged batches; we right-pad each wrapped sequence
        (flank_l + middle + flank_r) to a common length so one forward pass covers
        the chunk. Padding sits after flank_r, beyond the donor site's receptive
        field, so it never affects the two output positions read. The acceptor is
        output position 0; the donor is output position ``len(middle)+3``, which
        shifts per string -- hence the per-row gather rather than a fixed index.
        """
        lens = np.array([len(s) for s in strings])
        flank = len(self._flank_l) + len(self._flank_r)
        max_total = int(flank + lens.max())
        arr = np.zeros((len(strings), max_total), dtype=np.int64)
        for i, s in enumerate(strings):
            full = np.concatenate(
                [self._flank_l, np.asarray(s, np.int64), self._flank_r]
            )
            arr[i, : len(full)] = full
        with torch.no_grad():
            lyp = batched_run(self._model, arr).log_softmax(-1)
            rows = torch.arange(len(strings), device=lyp.device)
            don_pos = torch.tensor(lens + 3, device=lyp.device)
            acc = lyp[rows, 0, 1]
            don = lyp[rows, don_pos, 2]
            return torch.stack([acc, don], -1).mean(-1).cpu().numpy()

    def membership_queries(self, strings: List[List[int]], chunk=8192) -> np.ndarray:
        out = np.empty(len(strings), dtype=bool)
        for i in range(0, len(strings), chunk):
            score = self._raw_scores(strings[i : i + chunk])
            out[i : i + chunk] = (score - self._mean) / self._std > 0
        return out

    def membership_query(self, string: List[int]) -> bool:
        return bool(self.membership_queries([string])[0])


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
        # calib_len only avoids the permacached calibrate(); we use _base for scores.
        self._base = SpliceModelOracle(exon, model, calib_len=ref_len)
        self._n_max = n_max
        self._length = exon.random_text_length
        rng = np.random.default_rng(seed)
        mids = rng.integers(0, 4, size=(fit_count, ref_len)).tolist()
        S = self._base._raw_scores(mids).astype(np.float64)
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
        raw = self._base._raw_scores(strings).astype(np.float64)
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
        self._base = SpliceModelOracle(exon, model, calib_len=len_lo)
        self._n_max = n_max
        self._length = exon.random_text_length
        self._edges = np.arange(len_lo, len_hi + bin_width, bin_width)
        rng = np.random.default_rng(seed)
        self._bins = []  # (lo, hi, Fmean, Smean, beta, thresh)
        r2s = []
        for lo, hi in zip(self._edges[:-1], self._edges[1:]):
            lens = rng.integers(lo, hi, size=per_bin)
            mids = [rng.integers(0, 4, size=int(L)).tolist() for L in lens]
            S = self._base._raw_scores(mids).astype(np.float64)
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
                raw = self._base._raw_scores(sub).astype(np.float64)
                F = bow_features(sub, self._n_max).astype(np.float64)
                r = raw - (Sm + (F - Fm) @ beta)
                for k, i in enumerate(subidx):
                    out[i] = r[k] > th
        return out

    def membership_query(self, string):
        return bool(self.membership_queries([string])[0])


class SetDifferenceOracle(Oracle):
    """``a \\ b``: membership = ``a`` accepts AND ``b`` does not.

    Used to contrast SpliceAI against the fixed-motif model, e.g. with two balanced
    :class:`SpliceModelOracle`s (plain) or two :class:`CompositionResidualOracle`s
    (composition stripped from both before differencing).
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
