from typing import Callable, List

import numpy as np
import torch
import torch.nn.functional as F

from orthogonal_dfa.data.exon import RawExon
from orthogonal_dfa.l_star.structures import Oracle

# (model output tensor, middle lengths tensor) -> accept mask bool tensor
Readout = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


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


class SpliceModelOracle(Oracle):
    r"""E-L\* oracle that wraps/one-hots/batches queries and runs ``model`` (eval, no_grad, on ``device``), leaving the accept decision to ``readout``."""

    def __init__(
        self, exon: RawExon, model, readout: Readout, *, device=None, chunk: int = 1024
    ):
        model.eval()
        self._model = model
        self._readout = readout
        self._device = (
            torch.device(device)
            if device is not None
            else next(model.parameters()).device
        )
        trim = exon.cl // 2 + 2
        self._flank_l = np.array(exon.text[:trim], dtype=np.int64)
        self._flank_r = np.array(exon.text[-trim:], dtype=np.int64)
        self._length = exon.random_text_length
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
            wrapped, lengths = wrap_with_flanks(
                self._flank_l, self._flank_r, strings[i : i + self._chunk]
            )
            x = F.one_hot(torch.as_tensor(wrapped, device=self._device), 4).float()
            lens = torch.as_tensor(lengths, device=self._device)
            with torch.no_grad():
                pred = self._readout(self._model(x), lens)
            out[i : i + self._chunk] = pred.cpu().numpy().astype(bool)
        return out

    def membership_query(self, string: List[int]) -> bool:
        return bool(self.membership_queries([string])[0])


def spliceai_exon_scores(logits: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Exon score per sequence: mean of the acceptor logit at output position 0 and the donor logit at len(middle)+3."""
    lyp = logits.log_softmax(-1)
    rows = torch.arange(len(lyp), device=lyp.device)
    acc = lyp[rows, 0, 1]
    don = lyp[rows, lengths + 3, 2]
    return torch.stack([acc, don], -1).mean(-1)


def calibrated_spliceai_readout(threshold: float) -> Readout:
    """Readout that accepts when the SpliceAI exon score exceeds ``threshold``."""
    return lambda logits, lengths: spliceai_exon_scores(logits, lengths) > threshold


def median_threshold(
    model,
    exon: RawExon,
    length: int,
    *,
    count=20000,
    seed=0xCA11B,
    device=None,
    chunk=1024
):
    """Median exon score over ``count`` random length-``length`` middles (per-length since the score drifts with length)."""
    model.eval()
    device = (
        torch.device(device) if device is not None else next(model.parameters()).device
    )
    trim = exon.cl // 2 + 2
    flank_l = np.array(exon.text[:trim], dtype=np.int64)
    flank_r = np.array(exon.text[-trim:], dtype=np.int64)
    rng = np.random.default_rng(seed)
    scores = []
    for i in range(0, count, chunk):
        mids = rng.integers(0, 4, size=(min(chunk, count - i), length)).tolist()
        wrapped, lengths = wrap_with_flanks(flank_l, flank_r, mids)
        x = F.one_hot(torch.as_tensor(wrapped, device=device), 4).float()
        lens = torch.as_tensor(lengths, device=device)
        with torch.no_grad():
            scores.append(spliceai_exon_scores(model(x), lens).cpu().numpy())
    return float(np.median(np.concatenate(scores)))
