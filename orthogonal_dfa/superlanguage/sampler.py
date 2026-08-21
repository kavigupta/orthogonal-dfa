"""Sampling super-strings for the learner.

:class:`SuperSampler` is the super-alphabet counterpart of
:class:`~orthogonal_dfa.l_star.sampler.UniformSampler`: it draws E-L*'s probe
strings.  A super-string is drawn by generating a uniform random base stream and
:meth:`~orthogonal_dfa.superlanguage.vocabulary.KmerVocabulary.parse`-ing it until
``length`` super-symbols have been read, so the compiled base strings are exactly
uniform.  This is a Markov process, not i.i.d.: its per-symbol marginal is close
to but not equal to the fresh-position ``probabilities()``.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from orthogonal_dfa.l_star.sampler import Sampler

from .vocabulary import KmerVocabulary


@dataclass(frozen=True)
class SuperSampler(Sampler):
    vocabulary: KmerVocabulary
    length: int

    def sample(self, rng: np.random.Generator, alphabet_size: int) -> List[int]:
        assert alphabet_size == self.vocabulary.alphabet_size, (
            f"alphabet size mismatch: the vocabulary has "
            f"{self.vocabulary.alphabet_size} super-symbols but the learner asked "
            f"for {alphabet_size}"
        )
        vocab = self.vocabulary
        # Each super-symbol eats at most max_kmer_length base symbols, so this many
        # always parses to at least `length` of them; the tail is discarded.
        stream = rng.integers(
            vocab.base_alphabet_size, size=self.length * vocab.max_kmer_length
        )
        parsed = vocab.parse(stream.tolist())[: self.length]
        # parse emits the canonical wildcard; spread neutral positions over all of
        # them, which is what makes wildcard-only suffixes plentiful.
        wildcards = rng.integers(vocab.num_wildcards, size=self.length)
        return [
            vocab.num_kmers + int(w) if vocab.is_unknown(symbol) else symbol
            for symbol, w in zip(parsed, wildcards)
        ]
