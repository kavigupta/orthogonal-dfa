r"""An E-L\* membership oracle backed by a splice model (e.g. SpliceAI).

:class:`SpliceModelOracle` is only a **batching / format adapter**: it takes E-L\*'s
ragged ``List[List[int]]`` queries (variable-length middles over {A,C,G,T}), wraps
each with the exon's fixed flanks, right-pads the batch to a rectangle, chunks it,
and returns a numpy bool array. It does no model evaluation and no calibration of
its own -- those live in the ``scorer`` you pass in.

A ``scorer`` is any callable ``(wrapped, lengths) -> np.ndarray[bool]`` where
``wrapped`` is the padded ``(n, width)`` int array of flank+middle+flank sequences
and ``lengths`` is the ``(n,)`` array of middle lengths. Build a calibrated SpliceAI
scorer with :func:`calibrated_spliceai_scorer` (the calibration -- i.e. the accept
threshold -- is computed outside the oracle, e.g. via :func:`median_threshold`)::

    model = load_spliceai(400, 0)
    threshold = median_threshold(model, exon, length=95)   # calibrate OUTSIDE
    scorer = calibrated_spliceai_scorer(model, threshold)  # model + calibration
    oracle = SpliceModelOracle(exon, scorer)               # just batching
"""

from typing import Callable, List

import numpy as np

from orthogonal_dfa.data.exon import RawExon
from orthogonal_dfa.l_star.structures import Oracle

# A scorer maps a padded batch of wrapped sequences + their middle lengths to a
# per-sequence hard accept/reject call.
Scorer = Callable[[np.ndarray, np.ndarray], np.ndarray]


def wrap_with_flanks(flank_l, flank_r, strings):
    """Right-pad a ragged batch of middles wrapped in the flanks to a rectangle.

    Returns ``(wrapped, lengths)`` where ``wrapped`` is ``(n, width)`` int64 with
    each row ``flank_l + middle + flank_r`` followed by ``0`` padding, and
    ``lengths`` is the ``(n,)`` array of middle lengths. Padding sits after
    ``flank_r`` so a convolutional splice model's boundary outputs are unaffected.
    """
    lengths = np.array([len(s) for s in strings], dtype=np.int64)
    flank = len(flank_l) + len(flank_r)
    width = int(flank + (lengths.max() if len(strings) else 0))
    wrapped = np.zeros((len(strings), width), dtype=np.int64)
    for i, s in enumerate(strings):
        row = np.concatenate([flank_l, np.asarray(s, dtype=np.int64), flank_r])
        wrapped[i, : len(row)] = row
    return wrapped, lengths


class SpliceModelOracle(Oracle):
    r"""Batching/format adapter turning a calibrated ``scorer`` into an E-L\* oracle.

    Wraps each queried middle with the exon flanks, right-pads ragged batches,
    chunks, and converts to/from numpy -- nothing else. All model evaluation and
    calibration lives in ``scorer`` (see the module docstring).
    """

    def __init__(self, exon: RawExon, scorer: Scorer, *, chunk: int = 8192):
        trim = exon.cl // 2 + 2
        self._flank_l = np.array(exon.text[:trim], dtype=np.int64)
        self._flank_r = np.array(exon.text[-trim:], dtype=np.int64)
        self._length = exon.random_text_length
        self._scorer = scorer
        self._chunk = chunk

    @property
    def alphabet_size(self) -> int:
        return 4

    @property
    def string_length(self) -> int:
        return self._length

    def membership_queries(self, strings: List[List[int]]) -> np.ndarray:
        out = np.empty(len(strings), dtype=bool)
        for i in range(0, len(strings), self._chunk):
            batch = strings[i : i + self._chunk]
            wrapped, lengths = wrap_with_flanks(self._flank_l, self._flank_r, batch)
            out[i : i + self._chunk] = np.asarray(
                self._scorer(wrapped, lengths), dtype=bool
            )
        return out

    def membership_query(self, string: List[int]) -> bool:
        return bool(self.membership_queries([string])[0])


# -- SpliceAI scorer + calibration (external to the oracle) --------------------


def spliceai_exon_scores(model, wrapped: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Continuous exon score for each wrapped sequence: the mean of the acceptor
    logit at the 5' boundary (output position 0) and the donor logit at the 3'
    boundary (output position ``len(middle)+3``). Mirrors
    :func:`orthogonal_dfa.oracle.run_model.compute_exon_scores`, but gathers the
    donor position per row (it shifts with the middle length in a ragged batch).
    """
    import torch

    from orthogonal_dfa.oracle.run_model import batched_run

    with torch.no_grad():
        lyp = batched_run(model, wrapped).log_softmax(-1)
        rows = torch.arange(len(wrapped), device=lyp.device)
        don_pos = torch.tensor(lengths + 3, device=lyp.device)
        acc = lyp[rows, 0, 1]
        don = lyp[rows, don_pos, 2]
        return torch.stack([acc, don], -1).mean(-1).cpu().numpy()


def calibrated_spliceai_scorer(model, threshold: float) -> Scorer:
    """A scorer with calibration built in: SpliceAI exon score > ``threshold``."""

    def scorer(wrapped, lengths):
        return spliceai_exon_scores(model, wrapped, lengths) > threshold

    return scorer


def median_threshold(model, exon: RawExon, length: int, *, count=20000, seed=0xCA11B):
    """Calibration (done outside the oracle): the median exon score over ``count``
    random length-``length`` middles, so a scorer thresholding above it accepts
    ~50%. The exon score drifts with middle length, hence calibrating per length.
    """
    rng = np.random.default_rng(seed)
    mids = rng.integers(0, 4, size=(count, length)).tolist()
    trim = exon.cl // 2 + 2
    wrapped, lengths = wrap_with_flanks(
        np.array(exon.text[:trim], dtype=np.int64),
        np.array(exon.text[-trim:], dtype=np.int64),
        mids,
    )
    return float(np.median(spliceai_exon_scores(model, wrapped, lengths)))
