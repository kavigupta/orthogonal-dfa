import numpy as np
import pythomata
import torch
from automata.fa.dfa import DFA
from permacache import stable_hash
from torch import nn


class TorchDFA(nn.Module):
    def __init__(
        self,
        initial_state: int,
        accepting_states: torch.tensor,
        transition_function: torch.tensor,
    ):
        """

        :param initial_state: tensor, shape (num_dfas,)
        :param accepting_states: bool tensor, shape (num_dfas, num_states)
        :param transition_function: int tensor, shape (num_dfas, num_states, alphabet_size)
        """
        n_dfas, n_states, _ = transition_function.shape
        assert initial_state.shape == (n_dfas,)
        assert accepting_states.shape == (n_dfas, n_states)
        super().__init__()
        self.initial_state = nn.Parameter(initial_state, requires_grad=False)
        self.accepting_states = nn.Parameter(accepting_states, requires_grad=False)
        self.transition_function = nn.Parameter(
            transition_function, requires_grad=False
        )

    def __len__(self):
        return self.transition_function.shape[0]

    @property
    def alphabet_size(self):
        return self.transition_function.shape[2]

    @classmethod
    def none(cls, num_symbols):
        return cls.concat(num_symbols=num_symbols)

    @classmethod
    def random(cls, num_dfas, num_states, num_symbols, *, rng):
        transition_function = rng.integers(
            0, num_states, size=(num_dfas, num_states, num_symbols)
        )
        accepting_states = rng.random((num_dfas, num_states)) < 0.5
        initial_state = np.zeros((num_dfas,), dtype=np.int64)
        return cls(
            initial_state=torch.tensor(initial_state, dtype=torch.long),
            accepting_states=torch.tensor(accepting_states, dtype=torch.bool),
            transition_function=torch.tensor(transition_function, dtype=torch.long),
        )

    def __getitem__(self, idx):
        assert isinstance(idx, slice) or (
            isinstance(idx, np.ndarray) and idx.dtype == np.long
        )
        return TorchDFA(
            initial_state=self.initial_state[idx],
            accepting_states=self.accepting_states[idx],
            transition_function=self.transition_function[idx],
        )

    @classmethod
    def concat(cls, *dfas: "TorchDFA", num_symbols=None) -> "TorchDFA":
        if not dfas:
            assert num_symbols is not None
            return cls(
                initial_state=torch.zeros((0,), dtype=torch.long),
                accepting_states=torch.zeros((0, 0), dtype=torch.bool),
                transition_function=torch.zeros((0, 0, num_symbols), dtype=torch.long),
            )

        max_states = max((dfa.transition_function.shape[1] for dfa in dfas))
        transition_functions = []
        accepting_statess = []
        initial_states = []
        for dfa in dfas:
            additional_states = max_states - dfa.transition_function.shape[1]
            transition_function = torch.nn.functional.pad(
                dfa.transition_function, (0, 0, 0, additional_states), value=-1
            )
            accepting_states = torch.nn.functional.pad(
                dfa.accepting_states, (0, additional_states), value=False
            )
            transition_functions.append(transition_function)
            accepting_statess.append(accepting_states)
            initial_states.append(dfa.initial_state)
        transition_function = torch.cat(transition_functions, dim=0)
        accepting_states = torch.cat(accepting_statess, dim=0)
        initial_state = torch.cat(initial_states, dim=0)
        return cls(
            initial_state=initial_state,
            accepting_states=accepting_states,
            transition_function=transition_function,
        )

    @classmethod
    def from_pythomata(cls, dfa: pythomata.SimpleDFA):
        transition_function, accepting_states, initial_state = cls._to_tensor(dfa)

        return cls(
            initial_state=initial_state[None],
            accepting_states=accepting_states[None],
            transition_function=transition_function[None],
        )

    @classmethod
    def _to_tensor(cls, dfa):
        state_to_idx = {state: i for i, state in enumerate(sorted(dfa.states))}
        alphabet = sorted(dfa.alphabet)
        transition_function = torch.tensor(
            [
                [
                    state_to_idx[dfa.transition_function[state][symbol]]
                    for symbol in alphabet
                ]
                for state in sorted(dfa.states)
            ],
            dtype=torch.long,
        )
        accepting_states = torch.tensor(
            [state in dfa.accepting_states for state in sorted(dfa.states)],
            dtype=torch.bool,
        )
        initial_state = torch.tensor(state_to_idx[dfa.initial_state])
        return transition_function, accepting_states, initial_state

    def forward(self, x: torch.tensor, batch_size=100_000) -> torch.tensor:
        """
        :param x: LongTensor of shape (batch_size, seq_len) with symbols as integers
        :return: BoolTensor of shape (num_dfas, batch_size) indicating acceptance
        """
        results = []
        for i in range(0, x.shape[0], batch_size):
            results.append(self._forward_on_batch(x[i : i + batch_size]))
        return torch.cat(results, dim=1)

    def _forward_on_batch(self, x):
        batch_size, seq_len = x.shape
        states = self.initial_state[:, None].repeat(1, batch_size)
        for t in range(seq_len):
            symbols = x[:, t]
            states = self.transition_function[
                torch.arange(states.shape[0])[:, None], states, symbols
            ]
        return self.accepting_states[torch.arange(states.shape[0])[:, None], states]

    def extract_dfa(self, idx) -> pythomata.SimpleDFA:
        assert 0 <= idx < len(self)
        transition_function = {
            state: {
                symbol: int(new_state)
                for symbol, new_state in zip(
                    range(self.transition_function.shape[2]),
                    self.transition_function[idx, state].tolist(),
                )
            }
            for state in range(self.transition_function.shape[1])
        }
        return pythomata.SimpleDFA(
            states=set(range(self.transition_function.shape[1])),
            alphabet=set(range(self.transition_function.shape[2])),
            transition_function=transition_function,
            initial_state=int(self.initial_state[idx]),
            accepting_states={
                state
                for state in range(self.transition_function.shape[1])
                if self.accepting_states[idx, state]
            },
        )


def rename_symbols(dfa: pythomata.SimpleDFA, mapping: dict) -> pythomata.SimpleDFA:
    assert set(dfa.alphabet).issubset(
        set(mapping.keys())
    ), f"DFA alphabet is {set(dfa.alphabet)}; mapping keys are {set(mapping.keys())}"
    new_alphabet = set(mapping.values())
    return pythomata.SimpleDFA(
        states=dfa.states,
        alphabet=new_alphabet,
        transition_function={
            state: {
                mapping[symbol]: new_state for symbol, new_state in transitions.items()
            }
            for state, transitions in dfa.transition_function.items()
        },
        initial_state=dfa.initial_state,
        accepting_states=dfa.accepting_states,
    )


def rename_states(dfa, state_mapping):
    return pythomata.SimpleDFA(
        states=set(state_mapping.values()),
        alphabet=dfa.alphabet,
        transition_function={
            state_mapping[state]: {
                symbol: state_mapping[next_state]
                for symbol, next_state in transitions.items()
            }
            for state, transitions in dfa.transition_function.items()
        },
        initial_state=state_mapping[dfa.initial_state],
        accepting_states={state_mapping[s] for s in dfa.accepting_states},
    )


def dfa_symbols_to_num(dfa: pythomata.SimpleDFA) -> pythomata.SimpleDFA:
    return rename_symbols(
        dfa, {"A": 0, "C": 1, "G": 2, "T": 3, "0": 0, "1": 1, "2": 2, "3": 3}
    )


def dfa_symbols_to_acgt(dfa: pythomata.SimpleDFA) -> pythomata.SimpleDFA:
    return rename_symbols(dfa, {0: "A", 1: "C", 2: "G", 3: "T"})


def canonicalize_states(dfa: pythomata.SimpleDFA) -> pythomata.SimpleDFA:
    states_in_order = []
    visited = set()
    queue = [dfa.initial_state]
    while queue:
        state = queue.pop(0)
        if state in visited:
            continue
        visited.add(state)
        states_in_order.append(state)
        for symbol in sorted(dfa.alphabet):
            next_state = dfa.transition_function[state][symbol]
            if next_state not in visited:
                queue.append(next_state)
    state_mapping = {old: new for new, old in enumerate(states_in_order)}
    return rename_states(dfa, state_mapping)


def hash_dfa(dfa: pythomata.SimpleDFA) -> str:
    return str(
        stable_hash(
            (
                sorted(dfa.states),
                sorted(dfa.alphabet),
                sorted(
                    (state, symbol, new_state)
                    for state, transitions in dfa.transition_function.items()
                    for symbol, new_state in transitions.items()
                ),
                dfa.initial_state,
                sorted(dfa.accepting_states),
            )
        )
    )


def p_to_al(dfa: pythomata.SimpleDFA) -> DFA:
    """Convert a pythomata DFA to an automata-lib DFA."""
    return DFA(
        states=dfa.states,
        input_symbols=dfa.alphabet,
        transitions=dfa.transition_function,
        initial_state=dfa.initial_state,
        final_states=dfa.accepting_states,
    )
