from typing import Union

import numpy as np
import pythomata
from permacache import permacache

from orthogonal_dfa.mutation.mutation import Mutation
from orthogonal_dfa.oracle.evaluate import multidimensional_confusion
from orthogonal_dfa.utils.dfa import TorchDFA, hash_dfa


@permacache(
    "orthogonal_dfa/experiments/sensitivity/sensitivity_analysis",
    key_function=dict(
        dfa=hash_dfa, control_dfas=lambda x: tuple(hash_dfa(d) for d in x)
    ),
)
def sensitivity_analysis(
    dfa: pythomata.SimpleDFA,
    control_dfas: list[pythomata.SimpleDFA],
    model,
    exon,
    mutation: Mutation,
    *,
    num_samples: Union[int, "all"],
    seed: int,
):
    dfa = TorchDFA.from_pythomata(dfa)
    control_dfas = TorchDFA.concat(
        *[TorchDFA.from_pythomata(d) for d in control_dfas],
        num_symbols=dfa.alphabet_size,
    )
    rng = np.random.default_rng(seed)
    mut_desc = (
        mutation.all_mutations(dfa)
        if num_samples == "all"
        else mutation.sample_mutations(dfa, num_samples, rng)
    )
    dfas = mutation.apply_mutations(dfa, mut_desc)
    confs = multidimensional_confusion(
        exon,
        dfas,
        control_dfas,
        model,
        count=100_000,
        seed=rng.integers(1 << 32),
    )
    return confs, mut_desc
