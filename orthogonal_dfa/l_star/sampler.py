from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np


class Sampler(ABC):
    #: Number of symbols per sampled string; the learner reads it for
    #: state-reaching sampling and counterexample sizing.
    length: int

    @abstractmethod
    def sample(self, rng: np.random.Generator, alphabet_size: int) -> List[int]:
        pass

    def symbol_weights(
        self, rng: np.random.Generator, alphabet_size: int, num_strings: int = 200
    ) -> List[float]:
        """How often this sampler puts each symbol at a position, up to scale.

        Sampling a string that reaches a given state walks the DFA against these,
        so a learner drawing from a skewed distribution asks its oracle about the
        strings it will actually meet.  Estimated rather than declared: a sampler
        owes callers strings and nothing else.  Positions are taken as independent,
        which holds for the samplers here.
        """
        counts = np.ones(alphabet_size)
        for _ in range(num_strings):
            np.add.at(counts, self.sample(rng, alphabet_size), 1)
        return (counts / counts.sum()).tolist()


@dataclass(frozen=True)
class UniformSampler(Sampler):
    length: int

    def sample(self, rng: np.random.Generator, alphabet_size: int) -> List[int]:
        return rng.integers(0, alphabet_size, size=self.length).tolist()

    def symbol_weights(self, rng, alphabet_size, num_strings=200):
        """Ones: even needs no draw to measure, and integer weights keep the path
        counts integers, which is what makes them a count of strings."""
        return [1] * alphabet_size
