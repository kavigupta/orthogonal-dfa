from typing import Callable, List

import numpy as np
import torch
import torch.nn.functional as F
from permacache import permacache, stable_hash

from orthogonal_dfa.data.exon import RawExon
from orthogonal_dfa.l_star.structures import Oracle

# (model output tensor, middle lengths tensor) -> per-sequence value tensor
Readout = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

# Each flank keeps this many bases beyond the model's cl/2 half-context (matching
# data.sample_text's trim_zone = cl//2 + 2), so the output spans len(middle) +
# 2*FLANK_MARGIN positions: the acceptor is the first, the donor the last.
FLANK_MARGIN = 2

CALIBRATION_SEED = int(stable_hash("calibration"), 16)


def flanks(exon: RawExon):
    """The (left, right) fixed flanks the middle is wrapped in, as int arrays."""
    trim = exon.cl // 2 + FLANK_MARGIN
    return (
        np.array(exon.text[:trim], dtype=np.int64),
        np.array(exon.text[-trim:], dtype=np.int64),
    )


def wrap_with_flanks(flank_l, flank_r, strings):
    """Wrap each middle as flank_l+middle+flank_r, right-pad to a rectangle, and return (wrapped, middle_lengths)."""
    lengths = np.array([len(s) for s in strings], dtype=np.int64)
    flank = len(flank_l) + len(flank_r)
    width = int(flank + (lengths.max() if len(strings) else 0))
    wrapped = np.zeros((len(strings), width), dtype=np.int64)
    for i, s in enumerate(strings):
        row = np.concatenate([flank_l, np.asarray(s, dtype=np.int64), flank_r])
        wrapped[i, : len(row)] = row
    return wrapped, lengths


def run_over_middles(model, flank_l, flank_r, strings, readout, *, device, chunk):
    """Flank-wrap, one-hot, and run ``model`` (no_grad) over ``strings`` in chunks, returning np.concatenate of ``readout(logits, lengths)``."""
    parts = []
    for i in range(0, len(strings), chunk):
        wrapped, lengths = wrap_with_flanks(flank_l, flank_r, strings[i : i + chunk])
        x = F.one_hot(torch.as_tensor(wrapped, device=device), 4).float()
        lens = torch.as_tensor(lengths, device=device)
        with torch.no_grad():
            parts.append(readout(model(x), lens).cpu().numpy())
    return np.concatenate(parts) if parts else np.empty(0)


def _device_of(model, device):
    return (
        torch.device(device) if device is not None else next(model.parameters()).device
    )


class SpliceModelOracle(Oracle):
    r"""E-L\* oracle that wraps/one-hots/batches queries and runs ``model`` (eval, no_grad, on ``device``), leaving the accept decision to ``readout``."""

    def __init__(
        self, exon: RawExon, model, readout: Readout, *, device=None, chunk: int = 1024
    ):
        model.eval()
        self._model = model
        self._readout = readout
        self._device = _device_of(model, device)
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


def spliceai_exon_scores(logits: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Exon score per sequence: mean of the acceptor logit at the first output position and the donor logit at the last."""
    lyp = logits.log_softmax(-1)
    rows = torch.arange(len(lyp), device=lyp.device)
    acc = lyp[rows, 0, 1]
    don = lyp[rows, lengths + 2 * FLANK_MARGIN - 1, 2]
    return torch.stack([acc, don], -1).mean(-1)


def calibrated_spliceai_readout(threshold: float) -> Readout:
    """Readout that accepts when the SpliceAI exon score exceeds ``threshold``."""
    return lambda logits, lengths: spliceai_exon_scores(logits, lengths) > threshold


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
    chunk=1024
):
    """Median exon score over ``count`` random length-``length`` middles (per-length since the score drifts with length)."""
    model.eval()
    flank_l, flank_r = flanks(exon)
    mids = np.random.default_rng(seed).integers(0, 4, size=(count, length)).tolist()
    scores = run_over_middles(
        model,
        flank_l,
        flank_r,
        mids,
        spliceai_exon_scores,
        device=_device_of(model, device),
        chunk=chunk,
    )
    return float(np.median(scores))
