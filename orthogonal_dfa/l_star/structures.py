from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple, Union

from automata.fa.dfa import DFA
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


@dataclass(frozen=True)
class LStarDFA:
    pt: PrefixTreeNode
    dt: DecisionTree

    def to_dfa(self, oracle, all_symbols: List[str]) -> DFA:
        prefix_tree = self.pt.all_transitions()
        exemplar_map = self.pt.exemplars()

        states_visited = set()
        states_fringe = [self.pt.state_idx]
        transitions = {}
        while states_fringe:
            state = states_fringe.pop()
            exemplar = exemplar_map[state]
            if state in states_visited:
                continue
            states_visited.add(state)

            for symbol in all_symbols:
                if (state, symbol) in prefix_tree:
                    dest = prefix_tree[(state, symbol)]
                else:
                    dest = self.dt.classify(exemplar + symbol, oracle)
                transitions[(state, symbol)] = dest
                states_fringe.append(dest)
                states_visited.add(dest)
        return DFA(
            states=states_visited,
            input_symbols=set(all_symbols),
            transitions={(s, sym): d for (s, sym), d in transitions.items()},
            initial_state=str(self.pt.state_idx),
            final_states={
                s for s in states_visited if oracle.membership_query(exemplar_map[s])
            },
        )

    def split_state(
        self,
        parent_state_idx: int,
        action: str,
        distinguishing_string: str,
        oracle: Oracle,
    ) -> "LStarDFA":
        new_state_idx = len(self.pt.exemplars())  # increment the state counter
        pt = self.pt.split_state(parent_state_idx, action, new_state_idx)
        new_state_exemplar = pt.exemplars()[new_state_idx]
        return LStarDFA(
            pt=pt,
            dt=self.dt.split_state(
                new_state_idx,
                distinguishing_string,
                new_state_idx,
                oracle,
            ),
        )
