from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple, Union

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


class DecisionTree(ABC):
    @abstractmethod
    def classify(self, string: str, oracle: Oracle) -> int:
        pass

    @property
    def num_states(self) -> int:
        states = list(self.collect_states())
        assert set(states) == set(range(len(states)))
        return len(states)

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


@dataclass
class TriPredicate:
    vs: List[List[int]]
    evidence_threshold: float

    def predict(self, x: List[int], oracle: Oracle) -> float:
        return np.mean([oracle.membership_query(x + v) for v in self.vs])

    def __call__(self, x: List[int], oracle: Oracle) -> Union[bool, None]:
        f = self.predict(x, oracle)
        if f > self.evidence_threshold:
            return True
        if f < 1 - self.evidence_threshold:
            return False
        return None

    def __hash__(self):
        return hash((tuple(tuple(v) for v in self.vs), self.evidence_threshold))


def flat_decision_tree_to_decision_tree(
    fdt: List[List[Tuple["TriPredicate", bool]]],
) -> DecisionTree:
    """
    Takes a flat decision tree (fdt), which is represented as a list of descriptors of leaves, each
    being a list of decisions made along the path from the root to the leaf
    (represented as a tuple (predicate, decision))),
    and converts it into a hierarchical DecisionTree structure.
    """
    if not fdt:
        raise ValueError("Flat decision tree cannot be empty")

    # let partial_tree be a dictionary mapping from paths to DecisionTree nodes
    partial_tree: Dict[Tuple[Tuple["TriPredicate", bool], ...], DecisionTree] = {
        tuple(path): DecisionTreeLeafNode(i) for i, path in enumerate(fdt)
    }
    while len(partial_tree) > 1:
        # attempt to merge nodes that are the same except for the last decision
        path_1, path_2 = _locate_mergeable_paths(partial_tree)
        (*prefix, (predicate, is_accepting)) = path_1
        if is_accepting:
            path_1, path_2 = path_2, path_1
        node = DecisionTreeInternalNode(
            predicate=predicate,
            by_rejection=(
                partial_tree[path_1],
                partial_tree[path_2],
            ),
        )
        del partial_tree[path_1]
        del partial_tree[path_2]
        partial_tree[tuple(prefix)] = node
    return partial_tree[()]


def _locate_mergeable_paths(
    partial_tree: Dict[Tuple[Tuple["TriPredicate", bool], ...], DecisionTree],
) -> Tuple[
    Tuple[Tuple["TriPredicate", bool], ...], Tuple[Tuple["TriPredicate", bool], ...]
]:
    by_everything_but_last = defaultdict(list)
    for path in partial_tree.keys():
        by_everything_but_last[path[:-1]].append(path)
    assert any(len(v) >= 2 for v in by_everything_but_last.values())
    prefix = next(p for p, v in by_everything_but_last.items() if len(v) >= 2)
    assert len(by_everything_but_last[prefix]) == 2
    first, second = by_everything_but_last[prefix]
    assert first[-1][0] == second[-1][0]
    assert {first[-1][1], second[-1][1]} == {True, False}
    return first, second
