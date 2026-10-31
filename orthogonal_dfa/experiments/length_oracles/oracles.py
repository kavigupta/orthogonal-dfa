"""The two oracles: ``saw`` thresholds every string at the median for compiled length
40; ``flat`` thresholds each compiled length 35-85 at its own median."""

import numpy as np

from orthogonal_dfa.l_star.examples.gate_composition_residual import (
    fit_gate_composition_residual,
    gate_residual_oracle,
)
from orthogonal_dfa.l_star.examples.spliceai_oracle import (
    SpliceModelOracle,
    median_threshold,
)
from orthogonal_dfa.spliceai.exon_score import SpliceAIExonScore
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

#: Symbols 0-2 are the stops TAG, TAA, TGA; 3 and 4 are one-base wildcards.
VOCAB = KmerVocabulary(kmers=((3, 0, 2), (3, 0, 0), (3, 2, 0)), base_alphabet_size=4)
SYMBOLS = 36
LEN_LO, LEN_HI = 35, 85
SAW_LENGTH = 40


class PerLengthOracle(SpliceModelOracle):
    def __init__(self, exon, score_model, thresholds, **kw):
        super().__init__(exon, score_model, float("nan"), **kw)
        self.thresholds = dict(thresholds)
        self._lo = min(thresholds)
        self._thresholds = np.array([thresholds[l] for l in sorted(thresholds)])

    def membership_queries(self, strings):
        idx = np.clip(
            np.array([len(s) for s in strings]) - self._lo, 0, len(self._thresholds) - 1
        )
        return self.scores(strings) > self._thresholds[idx]


def oracle(kind, exon, model, device=None):
    if kind == "saw":
        return gate_residual_oracle(
            exon, model, length=SAW_LENGTH, device=device, len_lo=LEN_LO, len_hi=LEN_HI
        )
    assert kind == "flat", kind
    gscore = fit_gate_composition_residual(
        SpliceAIExonScore(model).eval(),
        exon,
        device=device,
        len_lo=LEN_LO,
        len_hi=LEN_HI,
    )
    thresholds = {
        l: median_threshold(gscore, exon, l, device=device)
        for l in range(LEN_LO, LEN_HI + 1)
    }
    return PerLengthOracle(exon, gscore, thresholds, device=device)
