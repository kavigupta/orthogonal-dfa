"""Small CPU SpliceAI fixtures shared by the composition-residual tests."""

import numpy as np
import torch

from orthogonal_dfa.data.exon import RawExon
from orthogonal_dfa.spliceai.exon_score import FLANK_MARGIN, SpliceAIExonScore
from orthogonal_dfa.spliceai.module import SpliceAIModule

SMALL_CL = 80
# RawExon.random_text_length = len(text) - (cl + 4), so this is the query length.
QUERY_LENGTH = 30


def small_module_and_exon(cl=SMALL_CL):
    torch.manual_seed(0)
    module = SpliceAIModule(window=cl)
    exon = RawExon(
        cl,
        np.random.default_rng(0)
        .integers(0, 4, size=cl + 2 * FLANK_MARGIN + QUERY_LENGTH)
        .tolist(),
    )
    return module, exon


def small_score_model_and_exon(cl=SMALL_CL):
    module, exon = small_module_and_exon(cl)
    return SpliceAIExonScore(module).eval(), exon
