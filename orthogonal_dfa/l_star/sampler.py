from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class Sampler(ABC):
    #: Number of symbols per sampled string; the learner reads it for
    #: state-reaching sampling and counterexample sizing.
    length: int

    @abstractmethod
    def sample(self, rng: np.random.Generator, alphabet_size: int) -> bytes:
        pass


@dataclass(frozen=True)
class UniformSampler(Sampler):
    length: int

    def sample(self, rng: np.random.Generator, alphabet_size: int) -> bytes:
        return rng.integers(
            0, alphabet_size, size=self.length, dtype=np.uint8
        ).tobytes()
