from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple, Union

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


class DecisionTree(ABC):
    @abstractmethod
    def classify(self, string: str, oracle: Oracle) -> int:
        pass

    @property
    def num_states(self) -> int:
        states = list(self.collect_states())
        assert set(states) == set(range(len(states)))
        return len(states)

    @property
    @abstractmethod
    def depth(self) -> int:
        pass

    @abstractmethod
    def collect_states(self) -> Iterable[int]:
        pass

    @abstractmethod
    def render(self, render_predicate, indent=0) -> List[str]:
        pass

    @abstractmethod
    def map_over_predicates(
        self,
        map_fn: Callable[
            [Callable[[str, Oracle], Union[bool, None]]],
            Callable[[str, Oracle], Union[bool, None]],
        ],
    ) -> "DecisionTree":
        pass


@dataclass(frozen=True)
class DecisionTreeInternalNode(DecisionTree):
    predicate: Callable[[str, Oracle], Union[bool, None]]
    by_rejection: Tuple[DecisionTree, DecisionTree]  # (if rejected, if accepted)

    def classify(self, string: str, oracle: Oracle) -> int:
        decision = self.predicate(string, oracle)
        if decision is None:
            return None
        decision = int(decision)
        return self.by_rejection[decision].classify(string, oracle)

    @property
    def depth(self) -> int:
        return 1 + max(child.depth for child in self.by_rejection)

    def collect_states(self) -> Iterable[int]:
        for child in self.by_rejection:
            yield from child.collect_states()

    def render(self, render_predicate, indent=0) -> List[str]:
        lines = []
        indent_str = " " * indent
        lines.append(f"{indent_str}{render_predicate(self.predicate)}:")
        lines += self.by_rejection[0].render(render_predicate, indent + 4)
        lines += self.by_rejection[1].render(render_predicate, indent + 4)
        return lines

    def map_over_predicates(
        self,
        map_fn: Callable[
            [Callable[[str, Oracle], Union[bool, None]]],
            Callable[[str, Oracle], Union[bool, None]],
        ],
    ) -> "DecisionTree":
        return DecisionTreeInternalNode(
            predicate=map_fn(self.predicate),
            by_rejection=tuple(
                child.map_over_predicates(map_fn) for child in self.by_rejection
            ),
        )


@dataclass(frozen=True)
class DecisionTreeLeafNode(DecisionTree):
    state_idx: int

    def classify(self, string: str, oracle: Oracle) -> int:
        del string, oracle  # unused
        return self.state_idx

    @property
    def depth(self) -> int:
        return 0

    def collect_states(self) -> Iterable[int]:
        yield self.state_idx

    def render(self, render_predicate, indent=0) -> List[str]:
        indent_str = " " * indent
        return [f"{indent_str}State {self.state_idx}"]

    def map_over_predicates(
        self,
        map_fn: Callable[
            [Callable[[str, Oracle], Union[bool, None]]],
            Callable[[str, Oracle], Union[bool, None]],
        ],
    ) -> "DecisionTree":
        return self


def classify_many(
    dt: DecisionTree, strings: List[List[int]], oracle: Oracle
) -> List[Union[int, None]]:
    """
    Equivalent of [dt.classify(s, oracle) for s in strings] but one batched oracle
    call per tree level.

    `dt` must be composed of TriPredicate nodes.
    """
    out = [None] * len(strings)
    node_of = [dt] * len(strings)  # each string's current node; None once resolved
    while True:
        # spans: [(index in strings, node, lo, hi)]; lo/hi are the slice of queries for this node
        queries, spans = [], []
        for i, node in enumerate(node_of):
            if node is None:
                continue
            if isinstance(node, DecisionTreeLeafNode):
                out[i], node_of[i] = node.state_idx, None
                continue
            assert isinstance(
                node.predicate, TriPredicate
            ), "classify_many needs TriPredicate nodes"
            lo = len(queries)
            queries.extend(strings[i] + v for v in node.predicate.vs)
            spans.append((i, node, lo, len(queries)))
        if not queries:
            return out
        answers = np.asarray(oracle.membership_queries(queries))
        assert len(answers) == len(queries), "oracle dropped answers"
        for i, node, lo, hi in spans:
            decision = node.predicate.decide(float(answers[lo:hi].mean()))
            # None => undecided: drop the string (out[i] stays None); else descend.
            node_of[i] = None if decision is None else node.by_rejection[int(decision)]


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
