from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from orthogonal_dfa.utils.dfa import TorchDFA


class Mutation(ABC):
    def perform_mutations(
        self, dfas: TorchDFA, num_mutations: int, rng: np.random.Generator
    ) -> Tuple[TorchDFA, np.ndarray]:
        mutation_reprs = self.sample_mutations(dfas, num_mutations, rng)
        dfas_expanded = self.apply_mutations(dfas, mutation_reprs)
        return (dfas_expanded, mutation_reprs)

    def _expand_to_number_mutations(self, dfas, num_mutations):
        initial_state = dfas.initial_state.repeat_interleave(num_mutations, dim=0)
        accepting_states = dfas.accepting_states.repeat_interleave(
            num_mutations, dim=0
        ).clone()
        transition_function = dfas.transition_function.repeat_interleave(
            num_mutations, dim=0
        ).clone()
        dfas_expanded = TorchDFA(
            initial_state=initial_state,
            accepting_states=accepting_states,
            transition_function=transition_function,
        )

        return dfas_expanded

    def apply_mutations(self, dfas: TorchDFA, mutations: np.ndarray) -> TorchDFA:
        """
        Takes a list of TorchDFAs and performs mutations.shape[0] // len(dfas) on each of them,
        returning a new list of mutated TorchDFAs.

        The output DFAs are flat, with batch axis `mutations.shape[0]`,
        but if reshaped would have shape `(len(dfas), mutations.shape[0] // len(dfas), ...)`.

        :param dfas: A batch of DFAs to mutate.
        :param mutations: an array represnenting the mutations. Either the output of `sample_mutations` or
            `all_mutations`.
        """
        assert (
            len(mutations) % len(dfas) == 0
        ), "Number of mutations must be a multiple of number of DFAs"
        dfas_expanded = self._expand_to_number_mutations(
            dfas, mutations.shape[0] // len(dfas)
        )
        self.apply_mutations_in_place(dfas_expanded, mutations)
        return dfas_expanded

    @abstractmethod
    def sample_mutations(
        self,
        dfas: TorchDFA,
        num_mutations: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def all_mutations(self, dfas: TorchDFA) -> np.ndarray:
        pass

    @abstractmethod
    def apply_mutations_in_place(self, dfas: TorchDFA, mutations: np.ndarray):
        pass


@dataclass
class RandomSingleMutation(Mutation):
    p_change_accept: float = 0.1

    def sample_mutations(
        self,
        dfas: TorchDFA,
        num_mutations: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        num_dfas, num_states, num_symbols = dfas.transition_function.shape
        num_mutations *= num_dfas
        representations = np.zeros((num_mutations, 4), dtype=np.uint32)
        is_change_accept = rng.random(num_mutations) < self.p_change_accept
        representations[is_change_accept, 0] = 0
        representations[is_change_accept, 1] = rng.integers(
            0, num_states, size=is_change_accept.sum()
        )
        representations[~is_change_accept, 0] = 1
        representations[~is_change_accept, 1] = rng.integers(
            0, num_states, size=(~is_change_accept).sum()
        )
        representations[~is_change_accept, 2] = rng.integers(
            0, num_symbols, size=(~is_change_accept).sum()
        )
        representations[~is_change_accept, 3] = rng.integers(
            0, num_states, size=(~is_change_accept).sum()
        )

        return representations

    def all_mutations(self, dfas: TorchDFA) -> np.ndarray:
        num_dfas, num_states, num_symbols = dfas.transition_function.shape

        representations_accept = np.zeros((num_dfas, num_states, 4), dtype=np.uint32)
        representations_accept[:, :, 0] = 0
        representations_accept[:, :, 1] = np.arange(num_states, dtype=np.uint32)
        representations_accept = representations_accept.reshape(-1, 4)
        representations_transition = np.zeros(
            (num_dfas, num_states, num_symbols, num_states, 4), dtype=np.uint32
        )
        representations_transition[:, :, :, :, 0] = 1
        representations_transition[:, :, :, :, 1] = np.arange(
            num_states, dtype=np.uint32
        )[:, None, None]
        representations_transition[:, :, :, :, 2] = np.arange(
            num_symbols, dtype=np.uint32
        )[None, :, None]
        representations_transition[:, :, :, :, 3] = np.arange(
            num_states, dtype=np.uint32
        )[None, None, :]
        representations_transition = representations_transition.reshape(-1, 4)

        representations = np.concatenate(
            [representations_accept, representations_transition], axis=0
        )

        return representations

    def apply_mutations_in_place(self, dfas: TorchDFA, mutations: np.ndarray):
        assert len(dfas) == len(mutations)
        is_accepting = mutations[:, 0] == 0
        accepting_muts = mutations[is_accepting]
        transition_muts = mutations[~is_accepting]

        dfas.accepting_states[is_accepting, accepting_muts[:, 1]] ^= True
        dfas.transition_function[
            ~is_accepting, transition_muts[:, 1], transition_muts[:, 2]
        ] = torch.tensor(
            transition_muts[:, 3],
            dtype=dfas.transition_function.dtype,
            device=dfas.transition_function.device,
        )


@dataclass
class RepeatedMutations(Mutation):
    base_mutation: Mutation
    num_serial: int

    def sample_mutations(
        self,
        dfas: TorchDFA,
        num_mutations: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return np.concatenate(
            [
                self.base_mutation.sample_mutations(dfas, num_mutations, rng)[..., None]
                for _ in range(self.num_serial)
            ],
            axis=-1,
        )

    def all_mutations(self, dfas: TorchDFA) -> np.ndarray:
        raise NotImplementedError(
            "Cannot enumerate all mutations when using RepeatedMutations"
        )

    def apply_mutations_in_place(self, dfas: TorchDFA, mutations: np.ndarray):
        assert mutations.shape[-1] == self.num_serial
        for i in range(self.num_serial):
            self.base_mutation.apply_mutations_in_place(dfas, mutations[..., i])
        return dfas
