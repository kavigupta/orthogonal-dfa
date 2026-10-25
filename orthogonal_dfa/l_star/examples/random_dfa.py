import itertools
import math
from typing import List, Optional

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


def _functional_graph_components(f: np.ndarray) -> np.ndarray:
    """A component label per state of the graph with edges ``q -> f[q]``."""
    label = np.arange(len(f))
    while True:
        new = np.minimum(label, label[f])
        np.minimum.at(new, f, new.copy())
        if (new == label).all():
            return label
        label = new


def sample_class_preserving_dfa(
    rng: np.random.Generator, *, num_states: int, alphabet_size: int, length: int
) -> Optional[DFA]:
    """
    Uniformly random transitions, with accepting states chosen so that a random
    length-``length`` string ``s`` preserves classes:

        q in F  <=>  delta(q, s) in F

    i.e. ``F`` is a union of components of ``delta(., s)``'s functional graph, the
    one whose acceptance rate at ``length`` is closest to 1/2. None if there is
    only one component, as there is when ``s`` synchronises the DFA.
    """
    table = rng.integers(num_states, size=(num_states, alphabet_size))
    s = rng.integers(alphabet_size, size=length)
    f = np.arange(num_states)
    for c in s:
        f = table[f, c]
    _, component = np.unique(_functional_graph_components(f), return_inverse=True)
    num_components = component.max() + 1
    if num_components == 1:
        return None

    step = np.zeros((num_states, num_states))
    for c in range(alphabet_size):
        np.add.at(step, (np.arange(num_states), table[:, c]), 1 / alphabet_size)
    mass = np.bincount(
        component,
        weights=np.linalg.matrix_power(step, length)[0],
        minlength=num_components,
    )
    accepting = min(
        (
            chosen
            for chosen in itertools.product([False, True], repeat=num_components)
            if 0 < sum(chosen) < num_components
        ),
        key=lambda chosen: abs(mass[list(chosen)].sum() - 0.5),
    )
    return DFA(
        states=set(range(num_states)),
        input_symbols=set(range(alphabet_size)),
        transitions={
            q: {c: int(table[q, c]) for c in range(alphabet_size)}
            for q in range(num_states)
        },
        initial_state=0,
        final_states={q for q in range(num_states) if accepting[component[q]]},
        allow_partial=False,
    )


def sample_minimal_admitted_dfa(
    rng: np.random.Generator, *, num_states: int, alphabet_size: int, length: int
) -> DFA:
    """
    A `sample_class_preserving_dfa`, minimised, with exactly ``num_states``
    states, that satisfies the preconditions at ``length``.
    """
    # minimisation shrinks these by about a quarter
    num_drawn = math.ceil(num_states * 4 / 3)
    while True:
        aut = sample_class_preserving_dfa(
            rng, num_states=num_drawn, alphabet_size=alphabet_size, length=length
        )
        if aut is None:
            continue
        aut = aut.minify()
        if len(aut.states) != num_states:
            continue
        if preconditions.satisfies_preconditions(aut, length=length):
            return aut
