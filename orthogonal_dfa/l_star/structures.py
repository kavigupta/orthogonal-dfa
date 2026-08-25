import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np


class NoiseModel(ABC):
    """Base class for noise models that add noise to oracle queries."""

    @abstractmethod
    def apply_noise(self, correct_value: bool, string: bytes, seed: int) -> bool:
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

    def apply_noise(self, correct_value: bool, string: bytes, seed: int) -> bool:
        def uniform_random(seed_obj: object) -> float:
            digest = hashlib.blake2b(repr(seed_obj).encode(), digest_size=8).digest()
            return int.from_bytes(digest, "big") / 2**64

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

    def apply_noise(self, correct_value: bool, string: bytes, seed: int) -> bool:
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
    def membership_query(self, string: bytes) -> bool:
        pass

    def membership_queries(self, strings: List[bytes]) -> np.ndarray:
        """
        Query multiple strings at once. Implementations can choose to override this
        and provide a more efficient batch query method. This does not have a cap
        on `strings`'s length.
        """
        return np.array([self.membership_query(s) for s in strings], dtype=bool)

    def target_dfa(self):
        """The DFA of the language this oracle answers for, over int symbols.

        ``None`` where there is no such DFA -- the language is not regular, or is
        a neural model we cannot write one for.  Answering is what lets a caller
        reason about the target's *states* rather than only its strings.
        """
        return None
