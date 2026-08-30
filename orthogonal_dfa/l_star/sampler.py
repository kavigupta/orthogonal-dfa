from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np


class Sampler(ABC):
    """Draws the strings a learner asks its oracle about.

    Implementations are hashed: the preconditions hold onto the sample a sampler
    draws rather than redrawing it per DFA.
    """

    #: Number of symbols per sampled string; the learner reads it for
    #: state-reaching sampling and counterexample sizing.
    length: int

    @abstractmethod
    def sample(self, rng: np.random.Generator, alphabet_size: int) -> List[int]:
        pass

    @abstractmethod
    def symbol_weights(self, alphabet_size: int) -> List[float]:
        """How often this sampler puts each symbol at a position, up to scale.

        Sampling a string that reaches a given state walks the DFA against these,
        so a learner drawing from a skewed distribution asks its oracle about the
        strings it will actually meet.  Only ratios are read, so any positive
        scaling of them says the same thing.  Positions are taken as independent.
        """


@dataclass(frozen=True)
class UniformSampler(Sampler):
    length: int

    def sample(self, rng: np.random.Generator, alphabet_size: int) -> List[int]:
        return rng.integers(0, alphabet_size, size=self.length).tolist()

    def symbol_weights(self, alphabet_size):
        return [1] * alphabet_size
