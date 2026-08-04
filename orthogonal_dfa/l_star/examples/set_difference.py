r"""The set-difference oracle and the fixed-motif (FM) model it contrasts SpliceAI against.

``a \ b`` accepts exactly the strings ``a`` accepts and ``b`` rejects, so running E-L\*
on it isolates what one splice model captures that another does not.  In practice that
is SpliceAI against the fixed-motif model loaded by :func:`load_fm`; both are wrapped
the same way (e.g. by ``SpliceAIExonScore`` into a ``SpliceModelOracle``) and differenced.
"""

import sys
from typing import List

import numpy as np

from orthogonal_dfa.l_star.structures import Oracle

# The fixed-motif ("FM") model is a trained modular_splicing model living in a separate
# repo on this machine (BothLSSIModels + an 82-motif RBNS PSAMMotifModel).
FM_REPO = "/mnt/md0/ExpeditionsCommon/spliceai/Canonical"
FM_MODEL_PREFIX = f"{FM_REPO}/model/msp-273.665a3"


def load_fm(seed=1):
    """Load the fixed-motif model (seed 1..5) as an eval/cuda nn.Module.

    ``modular_splicing`` lives in ``FM_REPO`` (added to ``sys.path`` here), not in this
    package's dependencies.  The returned model reads the same acceptor/donor logits as
    SpliceAI, so ``SpliceAIExonScore`` wraps it identically."""
    if FM_REPO not in sys.path:
        sys.path.insert(0, FM_REPO)
    from modular_splicing.utils.io import load_model  # pylint: disable=import-error

    _, model = load_model(f"{FM_MODEL_PREFIX}_{seed}")  # picks the latest step
    return model.eval().cuda()


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
