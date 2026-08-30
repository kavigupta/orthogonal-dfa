import hashlib
from typing import List

import numpy as np

from orthogonal_dfa.l_star.structures import Oracle

from .target import super_target_dfa
from .vocabulary import KmerVocabulary


def _compilation_seed(string: bytes, seed: int) -> int:
    """Deterministic in its arguments, so a query can be cached."""
    digest = hashlib.blake2b(
        repr((list(string), seed)).encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


class LiftedOracle(Oracle):
    """Labels a super-string by what one of its compilations is labelled.

    Which one is fixed by the string and the seed, so the answer is a function of
    the query.  That is the whole language only where the base oracle cannot see
    how the wildcards were filled -- where it can, ``super_target_dfa`` says so
    rather than there being an answer to average.
    """

    def __init__(
        self,
        base_oracle: Oracle,
        vocabulary: KmerVocabulary,
        *,
        seed: int,
    ):
        assert base_oracle.alphabet_size == vocabulary.base_alphabet_size, (
            f"base oracle alphabet ({base_oracle.alphabet_size}) does not match "
            f"the vocabulary's base alphabet ({vocabulary.base_alphabet_size})"
        )
        self._base = base_oracle
        self._vocab = vocabulary
        self._seed = seed

    @property
    def alphabet_size(self) -> int:
        return self._vocab.alphabet_size

    def membership_queries(self, strings: List[bytes]) -> np.ndarray:
        if not strings:
            return np.array([], dtype=bool)
        # One batch for all of them: the base oracle may be a model that is far
        # cheaper called once.
        rngs = [
            np.random.default_rng(_compilation_seed(string, self._seed))
            for string in strings
        ]
        # The vocabulary deals in symbol sequences; an oracle is asked in bytes.
        compiled = [bytes(c) for c in self._vocab.compile_many(strings, rngs)]
        labels = np.asarray(self._base.membership_queries(compiled), dtype=bool)
        assert labels.shape == (len(strings),), "base oracle dropped answers"
        return labels

    def membership_query(self, string: bytes) -> bool:
        return bool(self.membership_queries([string])[0])

    def target_dfa(self):
        base = self._base.target_dfa()
        return None if base is None else super_target_dfa(self._vocab, base)
