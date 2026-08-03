from typing import Callable, List

import numpy as np

from orthogonal_dfa.data.exon import RawExon
from orthogonal_dfa.l_star.structures import Oracle

# (wrapped [n, width] int, middle lengths [n]) -> accept mask [n] bool
Scorer = Callable[[np.ndarray, np.ndarray], np.ndarray]


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
    r"""E-L\* oracle that only batches: it wraps/pads/chunks queries and defers scoring to ``scorer``."""

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


def spliceai_exon_scores(model, wrapped: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Exon score per wrapped sequence: mean of the acceptor logit at output position 0 and the donor logit at len(middle)+3."""
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
    """Scorer that accepts when the SpliceAI exon score exceeds ``threshold``."""

    def scorer(wrapped, lengths):
        return spliceai_exon_scores(model, wrapped, lengths) > threshold

    return scorer


def median_threshold(model, exon: RawExon, length: int, *, count=20000, seed=0xCA11B):
    """Median exon score over ``count`` random length-``length`` middles (per-length since the score drifts with length)."""
    rng = np.random.default_rng(seed)
    mids = rng.integers(0, 4, size=(count, length)).tolist()
    trim = exon.cl // 2 + 2
    wrapped, lengths = wrap_with_flanks(
        np.array(exon.text[:trim], dtype=np.int64),
        np.array(exon.text[-trim:], dtype=np.int64),
        mids,
    )
    return float(np.median(spliceai_exon_scores(model, wrapped, lengths)))
