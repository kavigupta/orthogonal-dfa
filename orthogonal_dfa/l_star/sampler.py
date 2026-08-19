from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np


class Sampler(ABC):
    #: Number of symbols in each sampled string.  Part of the contract: the
    #: learner reads it for state-reaching sampling and counterexample sizing
    #: (see ``lstar`` and ``counterexample_synthesis``), so every Sampler must
    #: expose it.  Concrete samplers are dataclasses that supply it as a field.
    length: int

    @abstractmethod
    def sample(self, rng: np.random.Generator, alphabet_size: int) -> List[int]:
        pass


@dataclass(frozen=True)
class UniformSampler(Sampler):
    length: int

    def sample(self, rng: np.random.Generator, alphabet_size: int) -> List[int]:
        return rng.integers(0, alphabet_size, size=self.length).tolist()
