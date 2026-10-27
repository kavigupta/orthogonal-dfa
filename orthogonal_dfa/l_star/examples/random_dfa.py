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


_BATCH = 1024
_SCREEN_STRINGS = 1024


def dfa_from_table(table: np.ndarray, final: np.ndarray) -> DFA:
    """The DFA with ``table[q, c]`` the successor of ``q`` on ``c``, initial state 0."""
    num_states, alphabet_size = table.shape
    return DFA(
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


def sample_accessible_tables(
    rng: np.random.Generator, *, num_states: int, alphabet_size: int, count: int
) -> np.ndarray:
    """
    ``count`` transition tables, each uniform over those on ``num_states`` states
    in which every state is reachable from state 0, with ``tables[b, q, c]`` the
    successor of ``q`` on ``c``.
    """
    found = []
    while sum(len(t) for t in found) < count:
        tables = rng.integers(num_states, size=(count, num_states, alphabet_size))
        edge = (tables[:, :, :, None] == np.arange(num_states)).any(axis=2)
        reached = np.zeros((count, num_states), dtype=bool)
        reached[:, 0] = True
        for _ in range(num_states):
            reached |= (reached[:, :, None] & edge).any(axis=1)
        found.append(tables[reached.all(axis=1)])
    return np.concatenate(found)[:count]


def _class_pair_bound(tables: np.ndarray, finals: np.ndarray, length: int):
    """
    Per table, the least over pairs (p, q) of

        P_s[ delta(p, s) in class(p)  and  delta(q, s) in class(q) ]

    over random length-``length`` strings ``s``, an upper bound on the
    class-preserving fraction.
    """
    count, _, alphabet_size = tables.shape
    # (p, q) and (q, p) give the same probability, so a rejecting p and accepting
    # q need no chain of their own
    classes = [(False, False), (True, True), (True, False)]
    # kept[b, i, x, y]: the pair (x, y), required to end in classes[i]
    kept = np.stack(
        [(finals == a)[:, :, None] & (finals == b)[:, None, :] for a, b in classes],
        axis=1,
    ).astype(float)
    starts = kept.astype(bool)
    shape = kept.shape
    # successor[c]: flat index into kept of each entry's successor pair on c
    successor = [
        np.ravel_multi_index(
            np.broadcast_arrays(
                np.arange(count)[:, None, None, None],
                np.arange(len(classes))[None, :, None, None],
                tables[:, None, :, None, c],
                tables[:, None, None, :, c],
            ),
            shape,
        ).ravel()
        for c in range(alphabet_size)
    ]
    kept = kept.ravel()
    for _ in range(length):
        kept = sum(kept[index] for index in successor) / alphabet_size
    kept = kept.reshape(shape)
    return np.where(starts, kept, np.inf).min(axis=(1, 2, 3))


def _some_string_preserves(
    rng: np.random.Generator, tables: np.ndarray, finals: np.ndarray, *, length: int
) -> np.ndarray:
    """
    Per table, whether any of ``_SCREEN_STRINGS`` random strings preserves its
    classes. A DFA whose class-preserving fraction clears the bar fails this with
    probability

        (1 - MIN_CLASS_PRESERVING_FRAC) ** _SCREEN_STRINGS  ~  1e-9
    """
    count, num_states, alphabet_size = tables.shape
    strings = rng.integers(alphabet_size, size=(_SCREEN_STRINGS, length))
    # alive[b, i]: string i has kept the classes of every state run so far
    alive = np.ones((count, _SCREEN_STRINGS), dtype=bool)
    for q in range(num_states):
        table, string = np.nonzero(alive)
        ends = np.full(len(table), q)
        for j in range(length):
            ends = tables[table, ends, strings[string, j]]
        alive[table, string] = finals[table, ends] == finals[table, q]
    return alive.any(axis=1)


def _class_preserving_fraction(
    table: np.ndarray, final: np.ndarray, length: int
) -> float:
    """
    `preconditions.class_preserving_fraction`, over all 2^n subsets as bitmasks:

        h_0(S) = [S == F],   h_{t+1}(S) = mean_c h_t(pre_c(S)),   answer h_L(F)

    with ``pre_c(S) = {q : delta(q, c) in S}``.
    """
    num_states, alphabet_size = table.shape
    # pre[c, S], built up one state at a time
    pre = np.zeros((alphabet_size, 1), dtype=np.int64)
    for t in range(num_states):
        sources = ((table[:, :] == t) * (1 << np.arange(num_states))[:, None]).sum(0)
        pre = np.concatenate([pre, pre | sources[:, None]], axis=1)
    target = int((final * (1 << np.arange(num_states))).sum())
    h = np.zeros(1 << num_states)
    h[target] = 1.0
    for _ in range(length):
        h = sum(h[pre[c]] for c in range(alphabet_size)) / alphabet_size
    return float(h[target])


def sample_minimal_admitted_dfa(
    rng: np.random.Generator, *, num_states: int, alphabet_size: int, length: int
) -> DFA:
    """
    A `sample_accessible_tables` table with each state accepting with
    probability 1/2, kept if it is minimal and satisfies the preconditions at
    ``length``.

    Cheaper necessary conditions on the class-preserving fraction run first;
    all are exact but `_some_string_preserves`.
    """
    while True:
        tables = sample_accessible_tables(
            rng, num_states=num_states, alphabet_size=alphabet_size, count=_BATCH
        )
        finals = rng.random((_BATCH, num_states)) < 0.5
        keep = (
            _class_pair_bound(tables, finals, length)
            >= preconditions.MIN_CLASS_PRESERVING_FRAC
        )
        tables, finals = tables[keep], finals[keep]
        keep = _some_string_preserves(rng, tables, finals, length=length)
        for table, final in zip(tables[keep], finals[keep]):
            if (
                _class_preserving_fraction(table, final, length)
                < preconditions.MIN_CLASS_PRESERVING_FRAC
            ):
                continue
            aut = dfa_from_table(table, final)
            if len(aut.minify().states) != num_states:
                continue
            if preconditions.satisfies_preconditions(aut, length=length):
                return aut
