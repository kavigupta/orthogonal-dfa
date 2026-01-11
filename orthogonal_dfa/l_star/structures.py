from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple, Union

from frozendict import frozendict


class Oracle(ABC):
    @abstractmethod
    def membership_query(self, string: List[int]) -> bool:
        pass


@dataclass(frozen=True)
class PrefixTreeNode:
    state_idx: int
    children: frozendict[str, "PrefixTreeNode"]

    def all_transitions_iterable(self) -> Iterable[Tuple[str, str, str]]:
        for symbol, child in self.children.items():
            yield (self.state_idx, symbol, child.state_idx)
            yield from child.all_transitions_iterable()

    def all_transitions(self) -> Dict[Tuple[str, str], str]:
        return {
            (state, symbol): dest
            for state, symbol, dest in self.all_transitions_iterable()
        }

    def exemplars(self) -> Dict[int, str]:
        exemplars = {self.state_idx: ""}
        for symbol, child in self.children.items():
            for state_idx, exemplar in child.exemplars().items():
                exemplars[state_idx] = symbol + exemplar
        return exemplars

    def split_state(
        self, parent_state_idx: int, action: str, new_state_idx: int
    ) -> "PrefixTreeNode":
        if self.state_idx == parent_state_idx:
            assert action not in self.children, "Action already exists in children"
            new_child = PrefixTreeNode(
                state_idx=new_state_idx,
                children=frozendict(),
            )
            return PrefixTreeNode(
                state_idx=self.state_idx,
                children=frozendict({**self.children, action: new_child}),
            )
        new_children = {
            symbol: child.split_state(parent_state_idx, action, new_state_idx)
            for symbol, child in self.children.items()
        }
        return PrefixTreeNode(
            state_idx=self.state_idx,
            children=frozendict(new_children),
        )


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
    def split_state(
        self,
        current_state: int,
        new_state: int,
        predicate: Callable[[str, Oracle], Union[bool, None]],
    ) -> "DecisionTree":
        pass

    def add_state(
        self, to_split: int, predicate: Callable[[str, Oracle], bool]
    ) -> "DecisionTree":
        new_state = self.num_states
        return self.split_state(to_split, new_state, predicate)

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

    def split_state(
        self,
        current_state: int,
        new_state,
        predicate: Callable[[str, Oracle], bool],
    ) -> "DecisionTree":
        return DecisionTreeInternalNode(
            predicate=self.predicate,
            by_rejection=tuple(
                child.split_state(current_state, new_state, predicate)
                for child in self.by_rejection
            ),
        )

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

    def split_state(
        self,
        current_state: int,
        new_state: int,
        predicate: Callable[[str, Oracle], bool],
    ) -> "DecisionTree":
        if self.state_idx != current_state:
            return self
        return DecisionTreeInternalNode(
            predicate=predicate,
            by_rejection=(
                DecisionTreeLeafNode(state_idx=current_state),
                DecisionTreeLeafNode(state_idx=new_state),
            ),
        )

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
