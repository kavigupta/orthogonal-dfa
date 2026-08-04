import warnings
from typing import Callable, List

import numpy as np
import torch
from permacache import permacache, stable_hash

from orthogonal_dfa.data.exon import RawExon
from orthogonal_dfa.l_star.structures import Oracle
from orthogonal_dfa.spliceai.exon_score import (
    FLANK_MARGIN,
    device_of,
    forward_batch,
    spliceai_exon_scores,
)

# (model output tensor, middle lengths tensor) -> per-sequence value tensor
Readout = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

CALIBRATION_SEED = int(stable_hash("calibration"), 16)


def flanks(exon: RawExon):
    """The (left, right) fixed flanks the middle is wrapped in, as int arrays."""
    trim = exon.cl // 2 + FLANK_MARGIN
    return (
        np.array(exon.text[:trim], dtype=np.int64),
        np.array(exon.text[-trim:], dtype=np.int64),
    )


def wrap_with_flanks(flank_l, flank_r, strings):
    """Wrap each middle as flank_l+middle+flank_r, right-pad to a rectangle, and
    return (wrapped, middle_lengths).

    The pad value is arbitrary: a row's acceptor and donor readout positions sit
    cl/2 inside the row's own last real base, so neither one's receptive field
    ever reaches the padding, whatever the rest of the batch is.  That holds only
    with the model in eval mode -- BatchNorm over batch statistics would let the
    padded rows leak into every other row."""
    lengths = np.array([len(s) for s in strings], dtype=np.int64)
    flank = len(flank_l) + len(flank_r)
    width = int(flank + (lengths.max() if len(strings) else 0))
    wrapped = np.zeros((len(strings), width), dtype=np.int64)
    for i, s in enumerate(strings):
        row = np.concatenate([flank_l, np.asarray(s, dtype=np.int64), flank_r])
        wrapped[i, : len(row)] = row
    return wrapped, lengths


def run_over_middles(model, flank_l, flank_r, strings, readout, *, device, chunk):
    """Flank-wrap and run ``model`` over ``strings`` in chunks, returning
    np.concatenate of ``readout(logits, lengths)``."""
    parts = []
    for i in range(0, len(strings), chunk):
        wrapped, lengths = wrap_with_flanks(flank_l, flank_r, strings[i : i + chunk])
        logits = forward_batch(model, wrapped, device=device)
        lens = torch.as_tensor(lengths, device=device)
        parts.append(readout(logits, lens).cpu().numpy())
    return np.concatenate(parts) if parts else np.empty(0, dtype=bool)


class SpliceModelOracle(Oracle):
    r"""E-L\* oracle that wraps/one-hots/batches queries and runs ``model`` (under
    no_grad, on ``device``), deferring the accept decision to ``readout``.

    ``model`` must already be in eval mode -- forward_batch checks that on every
    query rather than this setting it, so a caller who shares the model with a
    training loop hears about it instead of having it silently switched.

    ``model`` is left on its own device unless ``device`` is passed, which only
    picks where the inputs go -- it does not move the model."""

    def __init__(
        self, exon: RawExon, model, readout: Readout, *, device=None, chunk: int = 1024
    ):
        self._model = model
        self._readout = readout
        self._device = device_of(model, device)
        self._flank_l, self._flank_r = flanks(exon)
        self._length = exon.random_text_length
        self._chunk = chunk

    @property
    def alphabet_size(self) -> int:
        return 4

    @property
    def string_length(self) -> int:
        return self._length

    def membership_queries(self, strings: List[List[int]]) -> np.ndarray:
        preds = run_over_middles(
            self._model,
            self._flank_l,
            self._flank_r,
            strings,
            self._readout,
            device=self._device,
            chunk=self._chunk,
        )
        return preds.astype(bool)

    def membership_query(self, string: List[int]) -> bool:
        return bool(self.membership_queries([string])[0])


class _CalibratedReadout:
    """Accepts when the exon score exceeds ``threshold``, warning when queried at
    a length ``threshold`` was not calibrated for.

    The score drifts with length, so a threshold is only near its intended accept
    rate close to its own calibration length -- one calibrated at 40 accepts ~77%
    of random length-80 middles.  E-L* queries a few lengths at once (2L for the
    probes, L..L+4 for the core), so this warns once per length rather than
    failing; until the threshold is per-length, expect the off-length rows to sit
    off 0.5."""

    def __init__(self, threshold: float, length: int):
        self.threshold = threshold
        self.calibration_length = length
        self._warned = set()

    def __call__(self, logits, lengths):
        for length in set(lengths.tolist()) - self._warned - {self.calibration_length}:
            self._warned.add(length)
            warnings.warn(
                f"threshold {self.threshold:.3f} was calibrated at middle length "
                f"{self.calibration_length} but is being applied at length {length}; "
                "the accept rate on these rows will be off its calibrated value"
            )
        return spliceai_exon_scores(logits, lengths) > self.threshold


def calibrated_spliceai_readout(threshold: float, length: int) -> Readout:
    """Readout that accepts when the SpliceAI exon score exceeds ``threshold``,
    which ``median_threshold`` computed at middle length ``length``."""
    return _CalibratedReadout(threshold, length)


@permacache(
    "orthogonal_dfa/l_star/examples/spliceai_oracle/median_threshold",
    key_function=dict(
        model=lambda m: stable_hash(m, version=2),
        exon=stable_hash,
        device=lambda _: None,  # does not affect the result
        chunk=lambda _: None,
    ),
)
def median_threshold(
    model,
    exon: RawExon,
    length: int,
    *,
    count=20000,
    seed=CALIBRATION_SEED,
    device=None,
    chunk=1024,
):
    """Median exon score over ``count`` random length-``length`` middles (per-length
    since the score drifts with length).

    oracle.run_model calibrates its dataset with a z-score thresholded at 0, i.e.
    at the *mean*; the median here instead puts the accept rate at exactly 0.5
    whatever the score distribution's skew, which is what E-L*'s degeneracy
    precondition wants.  The two conventions will disagree on borderline
    sequences, so do not mix a dataset built by one with an oracle built by the
    other.

    ``model`` must already be in eval mode; forward_batch checks it, which a
    permacache hit would skip along with the rest of the body."""
    flank_l, flank_r = flanks(exon)
    mids = np.random.default_rng(seed).integers(0, 4, size=(count, length)).tolist()
    scores = run_over_middles(
        model,
        flank_l,
        flank_r,
        mids,
        spliceai_exon_scores,
        device=device_of(model, device),
        chunk=chunk,
    )
    return float(np.median(scores))
