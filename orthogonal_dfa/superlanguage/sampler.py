from dataclasses import dataclass
from typing import List

import numpy as np

from orthogonal_dfa.l_star.sampler import Sampler

from .vocabulary import KmerVocabulary


@dataclass(frozen=True)
class SuperSampler(Sampler):
    """Draws by parsing a uniform base stream, which is what leaves the compiled
    strings uniform; drawing super-symbols independently would not.
    """

    vocabulary: KmerVocabulary
    length: int

    def sample(self, rng: np.random.Generator, alphabet_size: int) -> List[int]:
        assert alphabet_size == self.vocabulary.alphabet_size, (
            f"alphabet size mismatch: the vocabulary has "
            f"{self.vocabulary.alphabet_size} super-symbols but the learner asked "
            f"for {alphabet_size}"
        )
        vocab = self.vocabulary
        # A super-symbol eats at most one kmer's worth of base symbols, so this
        # many always parses to at least `length` of them; the tail is discarded.
        longest = max((len(k) for k in vocab.kmers), default=1)
        stream = rng.integers(vocab.base_alphabet_size, size=self.length * longest)
        parsed = vocab.parse(stream.tolist())[: self.length]
        # parse only ever emits the canonical wildcard. Spreading the neutral
        # positions over all of them is what makes such strings plentiful.
        wildcards = rng.integers(vocab.num_wildcards, size=self.length)
        return [
            vocab.num_kmers + int(w) if vocab.is_unknown(symbol) else symbol
            for symbol, w in zip(parsed, wildcards)
        ]
