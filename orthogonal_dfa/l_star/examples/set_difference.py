r"""The set-difference oracle and the fixed-motif (FM) model it contrasts SpliceAI against.

``a \ b`` accepts exactly the strings ``a`` accepts and ``b`` rejects, so running E-L\*
on it isolates what one splice model captures that another does not.  In practice that
is SpliceAI against the fixed-motif model loaded by :func:`load_fm`; both are wrapped
the same way (e.g. by ``SpliceAIExonScore`` into a ``SpliceModelOracle``) and differenced.
"""

import os
from typing import List

import numpy as np
import torch

from orthogonal_dfa.l_star.structures import Oracle

# Where the self-contained FM TorchScript traces live (gitignored; produced by
# scripts/convert_fm_to_torchscript.py on the machine that has the modular_splicing repo).
FM_TRACED_DIR = "data/pretrained_models"


def load_fm(seed=1):
    """Load the fixed-motif model (seed 1..5) as an eval/cuda TorchScript module.

    A trained modular_splicing model (BothLSSIModels + an 82-motif RBNS PSAMMotifModel),
    converted to a self-contained trace so loading needs only torch.  Reads the same
    acceptor/donor logits as SpliceAI, so ``SpliceAIExonScore`` wraps it identically."""
    path = os.path.join(FM_TRACED_DIR, f"fm-{seed}.traced.pt")
    assert os.path.exists(path), (
        f"{path} is missing; generate the FM traces (on the machine with the "
        f"modular_splicing repo) via scripts/convert_fm_to_torchscript.py"
    )
    return torch.jit.load(path).eval().cuda()


class SetDifferenceOracle(Oracle):
    r"""``a \ b``: accepts iff ``a`` accepts and ``b`` does not.

    A generic combinator over two oracles on the same alphabet; used to contrast
    SpliceAI against the FM (e.g. two balanced oracles, or two composition-residual
    oracles with composition stripped from both before differencing).
    """

    def __init__(self, oracle_a: Oracle, oracle_b: Oracle):
        assert (
            oracle_a.alphabet_size == oracle_b.alphabet_size
        ), "the two oracles must share an alphabet"
        self._a = oracle_a
        self._b = oracle_b

    @property
    def alphabet_size(self) -> int:
        return self._a.alphabet_size

    @property
    def string_length(self) -> int:
        return self._a.string_length

    def membership_queries(self, strings: List[List[int]]) -> np.ndarray:
        return self._a.membership_queries(strings) & ~self._b.membership_queries(
            strings
        )

    def membership_query(self, string: List[int]) -> bool:
        return bool(self.membership_queries([string])[0])
