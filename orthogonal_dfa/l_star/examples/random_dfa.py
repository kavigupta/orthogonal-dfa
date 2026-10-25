from typing import List

import numpy as np
from automata.fa.dfa import DFA

from orthogonal_dfa.l_star import preconditions


def sample_random_dfa(
    rng: np.random.Generator, *, num_states: int, alphabet_size: int = 2
) -> DFA:
    """
    A uniformly random complete DFA over symbols ``0..alphabet_size-1``.
    """
    transitions = {
        q: {c: int(rng.integers(num_states)) for c in range(alphabet_size)}
        for q in range(num_states)
    }
    final = {q for q in range(num_states) if rng.random() < 0.5}
    return DFA(
        states=set(range(num_states)),
        input_symbols=set(range(alphabet_size)),
        transitions=transitions,
        initial_state=0,
        final_states=final,
        allow_partial=False,
    )


def sample_admitted_dfas(
    rng: np.random.Generator,
    *,
    count: int,
    min_states: int,
    max_states: int,
    alphabet_size: int,
    length: int,
) -> List[DFA]:
    """
    The first ``count`` draws of `sample_random_dfa` that satisfy the
    preconditions at ``length``, each draw's state count uniform in
    ``[min_states, max_states]``.

    The state count is before minimisation; admitted DFAs are often smaller.
    """
    admitted = []
    while len(admitted) < count:
        num_states = int(rng.integers(min_states, max_states + 1))
        aut = sample_random_dfa(rng, num_states=num_states, alphabet_size=alphabet_size)
        if preconditions.satisfies_preconditions(aut, length=length):
            admitted.append(aut)
    return admitted
