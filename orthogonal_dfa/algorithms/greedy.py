from typing import List

import numpy as np
import pythomata
from permacache import permacache

from orthogonal_dfa.data.exon import RawExon
from orthogonal_dfa.mutation.mutation import Mutation
from orthogonal_dfa.oracle.evaluate import (
    ConditionalMutualInformation,
    multidimensional_confusion,
)
from orthogonal_dfa.utils.dfa import TorchDFA, hash_dfa


@permacache(
    "orthogonal_dfa/algorithms/greedy/greedy_optimize",
    key_function=dict(
        dfa=hash_dfa,
        control=lambda x: tuple(hash_dfa(d) for d in x),
    ),
)
def greedy_optimize(
    dfa: pythomata.SimpleDFA,
    exon: RawExon,
    model,
    mutation: Mutation,
    *,
    seed,
    scoring=ConditionalMutualInformation(),
    control: List[pythomata.SimpleDFA] = (),
    num_runs=100_000,
    tolerance=0.02,
):

    dfa = TorchDFA.from_pythomata(dfa)
    control = TorchDFA.concat(
        *[TorchDFA.from_pythomata(d) for d in control], num_symbols=dfa.alphabet_size
    )

    rng = np.random.default_rng(seed)
    path = []
    current_best_score = None
    while True:
        mut_desc = mutation.all_mutations(dfa)
        dfas = mutation.apply_mutations(dfa, mut_desc)
        confs = multidimensional_confusion(
            exon,
            TorchDFA.concat(dfas, dfa),
            control,
            model,
            count=num_runs,
            seed=rng.integers(1 << 32),
        )
        scores = scoring(confs)
        best_idx = int(np.argmax(scores))
        best_score = scores[best_idx].item()
        if current_best_score is None:
            current_best_score = scores[-1]
        if best_score <= current_best_score * (1 + tolerance):
            return dfa, path
        dfa = dfas[best_idx : best_idx + 1]
        current_best_score = best_score
        path.append((mut_desc[best_idx], best_score))
        print(best_idx, best_score)
