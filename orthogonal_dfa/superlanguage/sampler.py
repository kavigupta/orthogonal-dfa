"""Sampling super-strings for the learner.

:class:`SuperSampler` plays the role :class:`~orthogonal_dfa.l_star.sampler.UniformSampler`
plays for the base alphabet: it draws the random probe strings E-L* learns from.
Symbols are drawn i.i.d. from the vocabulary's emission distribution, and
``length`` is the number of *super-symbols* per string -- the same quantity the
learner's state-sampling and counterexample machinery read off ``sampler.length``.
The *compiled* base length varies from string to string (a kmer symbol expands to
several base symbols), which downstream base oracles handle.
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
        probs = self.vocabulary.probabilities()
        return rng.choice(len(probs), size=self.length, p=probs).tolist()
