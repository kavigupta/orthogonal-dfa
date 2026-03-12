from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np


class Sampler(ABC):
    @abstractmethod
    def sample(self, rng: np.random.Generator, alphabet_size: int) -> List[int]:
        pass


@dataclass(frozen=True)
class UniformSampler(Sampler):
    length: int

    def sample(self, rng: np.random.Generator, alphabet_size: int) -> List[int]:
        return rng.integers(0, alphabet_size, size=self.length).tolist()
