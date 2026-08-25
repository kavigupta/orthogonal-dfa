import hashlib
from typing import List, Optional

import numpy as np

from orthogonal_dfa.l_star.structures import NoiseModel, Oracle

from .vocabulary import KmerVocabulary


def _compilation_seed(string: List[int], seed: int, index: int) -> int:
    """Deterministic in its arguments, so a query can be cached."""
    digest = hashlib.blake2b(
        repr((list(string), seed, index)).encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


class LiftedOracle(Oracle):
    """Majority vote over several compilations, which only moves the answer for
    a base oracle that can see how the wildcards were filled.
    """

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
        # One batch for the whole cross product: the base oracle may be a model
        # that is far cheaper called once.
        repeated: List[List[int]] = []
        rngs: List[np.random.Generator] = []
        for string in strings:
            for j in range(self._num_compilations):
                repeated.append(string)
                rngs.append(
                    np.random.default_rng(_compilation_seed(string, self._seed, j))
                )
        flat = self._vocab.compile_many(repeated, rngs)
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
