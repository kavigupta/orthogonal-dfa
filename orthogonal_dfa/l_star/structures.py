from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union

import numpy as np


class NoiseModel(ABC):
    """Base class for noise models that add noise to oracle queries."""

    @abstractmethod
    def apply_noise(self, correct_value: bool, string: List[int], seed: int) -> bool:
        """
        Apply noise to a correct oracle value.

        Args:
            correct_value: The correct boolean value
            string: The input string being queried
            seed: Random seed for deterministic noise

        Returns:
            The noisy boolean value
        """


@dataclass(frozen=True)
class AsymmetricBernoulli(NoiseModel):
    """
    Asymmetric Bernoulli noise model.

    p_0: Probability of returning 1 (True) when the model output is 0 (False).
    p_1: Probability of returning 1 (True) when the model output is 1 (True).

    When correct_value is False: returns True with probability p_0, False with probability 1 - p_0.
    When correct_value is True: returns True with probability p_1, False with probability 1 - p_1.
    """

    p_0: float  # Probability of returning 1 when model output is 0
    p_1: float  # Probability of returning 1 when model output is 1

    def apply_noise(self, correct_value: bool, string: List[int], seed: int) -> bool:
        from permacache import stable_hash

        def uniform_random(seed_obj: object) -> float:
            hash_value = stable_hash(seed_obj)
            hash_value = (int(hash_value, 16) % 100) / 100
            return hash_value

        hash_input = uniform_random((string, seed))
        if correct_value:
            # When model output is 1, return 1 with probability p_1
            return hash_input < self.p_1
        # When model output is 0, return 1 with probability p_0
        return hash_input < self.p_0


@dataclass(frozen=True)
class SymmetricBernoulli(NoiseModel):
    """
    Symmetric Bernoulli noise model.

    With probability p_correct, returns the correct value.
    With probability 1 - p_correct, returns the flipped value.

    Implemented in terms of AsymmetricBernoulli with p_0 = 1 - p_correct and p_1 = p_correct.
    This satisfies: accuracy = p_1 = 1 - p_0 = p_correct.
    """

    p_correct: float

    def apply_noise(self, correct_value: bool, string: List[int], seed: int) -> bool:
        # Use AsymmetricBernoulli with p_0 = 1 - p_correct and p_1 = p_correct
        # This satisfies: accuracy = p_1 = 1 - p_0 = p_correct
        asymmetric = AsymmetricBernoulli(p_0=1 - self.p_correct, p_1=self.p_correct)
        return asymmetric.apply_noise(correct_value, string, seed)


class Oracle(ABC):
    @property
    @abstractmethod
    def alphabet_size(self) -> int:
        pass

    @abstractmethod
    def membership_query(self, string: List[int]) -> bool:
        pass

    def membership_queries(self, strings: List[List[int]]) -> np.ndarray:
        """
        Query multiple strings at once. Implementations can choose to override this
        and provide a more efficient batch query method. This does not have a cap
        on `strings`'s length.
        """
        return np.array([self.membership_query(s) for s in strings], dtype=bool)


@dataclass
class TriPredicate:
    vs: List[List[int]]
    accept_threshold: float
    reject_threshold: float

    def predict(self, x: List[int], oracle: Oracle) -> float:
        answers = oracle.membership_queries([x + v for v in self.vs])
        assert len(answers) == len(self.vs), "oracle dropped answers"
        return float(np.mean(answers))

    def decide(self, f: float) -> Union[bool, None]:
        if f > self.accept_threshold:
            return True
        if f < self.reject_threshold:
            return False
        return None

    def __call__(self, x: List[int], oracle: Oracle) -> Union[bool, None]:
        return self.decide(self.predict(x, oracle))

    def __hash__(self):
        return hash(
            (
                tuple(tuple(v) for v in self.vs),
                self.accept_threshold,
                self.reject_threshold,
            )
        )
