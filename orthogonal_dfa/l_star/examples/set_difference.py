r"""The set-difference oracle: ``a \ b`` accepts the strings ``a`` accepts and ``b``
rejects.

Running E-L\* on it isolates what one splice model captures that another does not -- in
practice SpliceAI against the fixed-motif model (``load_fm`` in
``orthogonal_dfa.spliceai.load_model``); both are wrapped the same way (e.g. by
``SpliceAIExonScore`` into a ``SpliceModelOracle``) and differenced.
"""

from typing import List

import numpy as np

from orthogonal_dfa.l_star.structures import Oracle


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
