from typing import List

import numpy as np
import torch
from permacache import permacache, stable_hash

from orthogonal_dfa.data.exon import RawExon
from orthogonal_dfa.l_star.structures import Oracle
from orthogonal_dfa.spliceai.exon_score import FLANK_MARGIN, device_of, one_hot

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


def run_over_middles(score_model, flank_l, flank_r, strings, *, device, chunk):
    """Flank-wrap, one-hot, and run ``score_model(x, lengths)`` (no_grad) over
    ``strings`` in chunks, returning the concatenated per-sequence scores."""
    # Checked on every call rather than once at setup: wrap_with_flanks's padding
    # only stays inert in eval mode, and a caller sharing the model with a training
    # loop can flip a submodule back into train mode at any point.
    assert not any(m.training for m in score_model.modules()), (
        "score_model must be in eval mode: in train mode BatchNorm normalizes over "
        "the batch, so padded rows leak into every other row's score"
    )
    parts = []
    for i in range(0, len(strings), chunk):
        wrapped, lengths = wrap_with_flanks(flank_l, flank_r, strings[i : i + chunk])
        x = one_hot(wrapped, device=device)
        lens = torch.as_tensor(lengths, device=device)
        with torch.no_grad():
            parts.append(score_model(x, lens).cpu().numpy())
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)


class SpliceModelOracle(Oracle):
    r"""E-L\* oracle that wraps/one-hots/batches queries, runs ``score_model(x,
    lengths)`` (no_grad, on ``device``), and accepts when the score exceeds
    ``threshold``.

    ``score_model`` must already be in eval mode -- run_over_middles checks that on
    every query rather than this setting it, so a caller who shares the model with a
    training loop hears about it instead of having it silently switched.  ``device``
    only picks where the inputs go; it does not move the model."""

    def __init__(
        self,
        exon: RawExon,
        score_model,
        threshold: float,
        *,
        device=None,
        chunk: int = 1024,
    ):
        self._score_model = score_model
        self._threshold = threshold
        self._device = device_of(score_model, device)
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
        scores = run_over_middles(
            self._score_model,
            self._flank_l,
            self._flank_r,
            strings,
            device=self._device,
            chunk=self._chunk,
        )
        return scores > self._threshold

    def membership_query(self, string: List[int]) -> bool:
        return bool(self.membership_queries([string])[0])


@permacache(
    "orthogonal_dfa/l_star/examples/spliceai_oracle/median_threshold",
    key_function=dict(
        score_model=lambda m: stable_hash(m, version=2),
        exon=stable_hash,
        device=lambda _: None,  # does not affect the result
        chunk=lambda _: None,
    ),
)
def median_threshold(
    score_model,
    exon: RawExon,
    length: int,
    *,
    count=20000,
    seed=CALIBRATION_SEED,
    device=None,
    chunk=1024,
):
    """Median exon score over ``count`` random length-``length`` middles (per-length
    since the score drifts with length), i.e. the threshold that makes
    ``SpliceModelOracle`` accept ~50% of length-``length`` middles.

    run_model calibrates its dataset with a z-score thresholded at 0 (the *mean*);
    the median here puts the accept rate at exactly 0.5 whatever the distribution's
    skew, so do not mix a dataset built by one with an oracle built by the other.

    ``score_model`` must already be in eval mode; a permacache hit skips the check
    along with the rest of the body."""
    flank_l, flank_r = flanks(exon)
    mids = np.random.default_rng(seed).integers(0, 4, size=(count, length)).tolist()
    scores = run_over_middles(
        score_model,
        flank_l,
        flank_r,
        mids,
        device=device_of(score_model, device),
        chunk=chunk,
    )
    return float(np.median(scores))
