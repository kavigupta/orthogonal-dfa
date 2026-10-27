import itertools
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


def sample_accessible_transitions(
    rng: np.random.Generator, *, num_states: int, alphabet_size: int
) -> np.ndarray:
    """
    Uniform over transition tables on ``num_states`` states in which every state
    is reachable from state 0, with ``table[q, c]`` the successor of ``q`` on ``c``.
    """
    while True:
        table = rng.integers(num_states, size=(num_states, alphabet_size))
        reached = np.zeros(num_states, dtype=bool)
        reached[0] = True
        while True:
            grown = reached.copy()
            grown[table[reached].ravel()] = True
            if (grown == reached).all():
                break
            reached = grown
        if reached.all():
            return table


def class_preserving_final_states(
    rng: np.random.Generator, table: np.ndarray, *, length: int
) -> Optional[np.ndarray]:
    """
    Accepting states chosen so that a random length-``length`` string ``s``
    preserves classes:

        q in F  <=>  delta(q, s) in F

    i.e. ``F`` is a union of components of ``delta(., s)``'s functional graph, the
    one whose acceptance rate at ``length`` is closest to 1/2. None if there is
    only one component, as there is when ``s`` synchronises the DFA.
    """
    num_states, alphabet_size = table.shape
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
    return np.array(accepting)[component]


def sample_minimal_admitted_dfa(
    rng: np.random.Generator, *, num_states: int, alphabet_size: int, length: int
) -> DFA:
    """
    A `sample_accessible_transitions` table with `class_preserving_final_states`,
    kept if it is minimal and satisfies the preconditions at ``length``.
    """
    while True:
        table = sample_accessible_transitions(
            rng, num_states=num_states, alphabet_size=alphabet_size
        )
        final = class_preserving_final_states(rng, table, length=length)
        if final is None:
            continue
        aut = DFA(
            states=set(range(num_states)),
            input_symbols=set(range(alphabet_size)),
            transitions={
                q: {c: int(table[q, c]) for c in range(alphabet_size)}
                for q in range(num_states)
            },
            initial_state=0,
            final_states={q for q in range(num_states) if final[q]},
            allow_partial=False,
        )
        if len(aut.minify().states) != num_states:
            continue
        if preconditions.satisfies_preconditions(aut, length=length):
            return aut
