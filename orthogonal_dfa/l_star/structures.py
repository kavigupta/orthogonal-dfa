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


def _uniform_random(string: bytes, seed: int) -> float:
    """A uniform draw on [0, 1) keyed by ``(string, seed)``.

    Hashes the string itself rather than its repr: this runs once per membership
    query, and formatting dominated it.
    """
    digest = hashlib.blake2b(
        bytes(string) + seed.to_bytes(8, "big", signed=True), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") / 2**64


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
        hash_input = _uniform_random(string, seed)
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

    The AsymmetricBernoulli with p_0 = 1 - p_correct and p_1 = p_correct, inlined:
    this runs once per membership query, and constructing that dataclass here cost
    more than the draw it wrapped.
    """

    p_correct: float

    def apply_noise(self, correct_value: bool, string: bytes, seed: int) -> bool:
        hash_input = _uniform_random(string, seed)
        if correct_value:
            return hash_input < self.p_correct
        return hash_input < 1 - self.p_correct


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

    @property
    def string_length(self) -> int:
        """The one length this oracle answers about, where it answers about one.

        Not every oracle is over strings of a fixed length -- a regex or a parity
        is over all of them -- so this raises rather than answering for those.  A
        caller reading it is one that needs the length, and an oracle that has
        none cannot serve it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} answers about strings of any length"
        )

    def target_dfa(self):
        """The DFA of the language this oracle answers for, over int symbols.

        ``None`` where there is no such DFA -- the language is not regular, or is
        a neural model we cannot write one for.  Answering is what lets a caller
        reason about the target's *states* rather than only its strings.
        """
        return None


@dataclass(frozen=True)
class NoisyOracle(Oracle):
    """``inner``'s answers with ``noise_model`` applied to each.

    An oracle that defines a language does not also have to own the noise on it,
    which is a property of how it is being read rather than of the language.  The
    target DFA passes through for the same reason: noise moves answers, not the
    language they are answers about.
    """

    inner: Oracle
    noise_model: NoiseModel
    seed: int

    @property
    def alphabet_size(self) -> int:
        return self.inner.alphabet_size

    @property
    def string_length(self) -> int:
        return self.inner.string_length

    def membership_query(self, string: List[int]) -> bool:
        return self.noise_model.apply_noise(
            self.inner.membership_query(string), string, self.seed
        )

    def membership_queries(self, strings: List[List[int]]) -> np.ndarray:
        answers = self.inner.membership_queries(strings)
        return np.array(
            [
                self.noise_model.apply_noise(bool(answer), string, self.seed)
                for answer, string in zip(answers, strings)
            ],
            dtype=bool,
        )

    def target_dfa(self):
        return self.inner.target_dfa()
