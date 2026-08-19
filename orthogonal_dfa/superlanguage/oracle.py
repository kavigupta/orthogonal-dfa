"""Lift a base-alphabet oracle to the super alphabet.

:class:`LiftedOracle` answers membership for super-strings by *compiling* each one
back to the base alphabet and asking a base oracle.  Because compilation is
nondeterministic (every ``X`` becomes an independent uniform base symbol), a
single super-string maps to a distribution over base strings; the oracle draws
``num_compilations`` of them, queries the base oracle on all of them, and returns
the majority vote.  Averaging the compilations is the "sample several strings and
take the mean to boost signal" step -- for a super-string whose base-label
probability sits away from 1/2 it sharpens the answer, at a cost linear in
``num_compilations``.

Compilation randomness is a deterministic function of the super-string and
``seed`` (hashed exactly the way the noise models in ``structures`` hash), so the
lifted oracle is a reproducible, memoizable function: querying the same
super-string twice gives the same answer, and different super-strings get
independent compilation draws.
"""

import hashlib
from typing import List, Optional

import numpy as np

from orthogonal_dfa.l_star.structures import NoiseModel, Oracle

from .vocabulary import KmerVocabulary


def _compilation_seed(string: List[int], seed: int, index: int) -> int:
    """A 64-bit seed for the ``index``-th compilation of ``string`` under
    ``seed``.  Deterministic in its inputs so the oracle is a pure function."""
    digest = hashlib.blake2b(
        repr((list(string), seed, index)).encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


class LiftedOracle(Oracle):
    def __init__(
        self,
        base_oracle: Oracle,
        vocabulary: KmerVocabulary,
        *,
        num_compilations: int = 8,
        seed: int = 0,
        noise_model: Optional[NoiseModel] = None,
    ):
        assert num_compilations >= 1, "need at least one compilation to vote on"
        assert base_oracle.alphabet_size == vocabulary.base_alphabet_size, (
            f"base oracle alphabet ({base_oracle.alphabet_size}) does not match "
            f"the vocabulary's base alphabet ({vocabulary.base_alphabet_size})"
        )
        self._base = base_oracle
        self._vocab = vocabulary
        self._num_compilations = num_compilations
        self._seed = seed
        self._noise_model = noise_model

    @property
    def alphabet_size(self) -> int:
        return self._vocab.alphabet_size

    def membership_queries(self, strings: List[List[int]]) -> np.ndarray:
        if not strings:
            return np.array([], dtype=bool)
        # Flatten (string, compilation) into a single base-oracle batch so the
        # base oracle -- which may be an expensive batched model -- is called once.
        flat: List[List[int]] = []
        for string in strings:
            for j in range(self._num_compilations):
                rng = np.random.default_rng(
                    _compilation_seed(string, self._seed, j)
                )
                flat.append(self._vocab.compile(string, rng))
        base = np.asarray(self._base.membership_queries(flat), dtype=bool)
        assert base.shape == (len(flat),), "base oracle dropped answers"
        votes = base.reshape(len(strings), self._num_compilations).mean(axis=1) >= 0.5
        if self._noise_model is not None:
            votes = np.array(
                [
                    self._noise_model.apply_noise(bool(v), s, self._seed)
                    for v, s in zip(votes, strings)
                ],
                dtype=bool,
            )
        return votes

    def membership_query(self, string: List[int]) -> bool:
        return bool(self.membership_queries([string])[0])
