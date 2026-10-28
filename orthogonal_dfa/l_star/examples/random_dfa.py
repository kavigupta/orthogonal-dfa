from typing import List

import numpy as np
from automata.fa.dfa import DFA
from numba import njit

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


# float64 rounding in the chain is ~1e-15; a fraction this close to the bar goes
# on to the exact check in preconditions
_SLACK = 1e-9
_CANDIDATES_PER_SEARCH = 1 << 20


def dfa_from_table(table: np.ndarray, final_mask: int) -> DFA:
    """The DFA with ``table[q, c]`` the successor of ``q`` on ``c``, initial state 0,
    and state ``q`` accepting iff bit ``q`` of ``final_mask`` is set."""
    num_states, alphabet_size = table.shape
    return DFA(
        states=set(range(num_states)),
        input_symbols=set(range(alphabet_size)),
        transitions={
            q: {c: int(table[q, c]) for c in range(alphabet_size)}
            for q in range(num_states)
        },
        initial_state=0,
        final_states={q for q in range(num_states) if (final_mask >> q) & 1},
        allow_partial=False,
    )


@njit(cache=True)
def _draw_accessible(table):
    """Fills ``table`` uniformly among those in which every state is reachable
    from state 0, using numba's generator."""
    num_states, alphabet_size = table.shape
    while True:
        for q in range(num_states):
            for c in range(alphabet_size):
                table[q, c] = np.random.randint(0, num_states)
        reached = 1
        while True:
            grown = reached
            for q in range(num_states):
                if (reached >> q) & 1:
                    for c in range(alphabet_size):
                        grown |= 1 << table[q, c]
            if grown == reached:
                break
            reached = grown
        if reached == (1 << num_states) - 1:
            return


@njit(cache=True)
def accessible_tables(seed, num_states, alphabet_size, count):
    """``count`` draws of `_draw_accessible` from numba's generator seeded with ``seed``."""
    np.random.seed(seed)
    tables = np.empty((count, num_states, alphabet_size), np.int64)
    for i in range(count):
        _draw_accessible(tables[i])
    return tables


@njit(cache=True)
def _is_minimal(table, final_mask):
    """Moore refinement; ``table`` is accessible, so minimal iff every state is
    its own class."""
    num_states, alphabet_size = table.shape
    cls = np.empty(num_states, np.int64)
    new = np.empty(num_states, np.int64)
    for q in range(num_states):
        cls[q] = (final_mask >> q) & 1
    num_classes = 2
    while True:
        new[:] = -1
        m = 0
        for q in range(num_states):
            if new[q] >= 0:
                continue
            new[q] = m
            for r in range(q + 1, num_states):
                if new[r] >= 0 or cls[r] != cls[q]:
                    continue
                same = True
                for c in range(alphabet_size):
                    if cls[table[r, c]] != cls[table[q, c]]:
                        same = False
                        break
                if same:
                    new[r] = m
            m += 1
        cls[:] = new
        if m in (num_classes, num_states):
            return m == num_states
        num_classes = m


@njit(cache=True)
def _workspace(num_states, alphabet_size):
    """Scratch for `_subsets_reachable` on tables of this shape, as
    ``(index, subsets, successor)``."""
    return (
        np.full(1 << num_states, -1, np.int64),
        np.empty(1 << num_states, np.int64),
        np.empty((1 << num_states, alphabet_size), np.int64),
    )


@njit(cache=True)
def _subsets_reachable(table, final_mask, workspace):
    """
    The subsets reachable from F under S -> pre_c(S) = {q : delta(q, c) in S},
    numbered in discovery order with F as 0, as ``successor[i, c]``: a view into
    ``workspace``, valid until its next use.
    """
    num_states, alphabet_size = table.shape
    # pre_c(S) as an OR of lookups on S's nibbles: nibble[c, i, v] is pre_c of
    # the states 4i + (bits of v)
    num_nibbles = (num_states + 3) // 4
    nibble = np.zeros((alphabet_size, num_nibbles, 16), np.int64)
    for c in range(alphabet_size):
        for q in range(num_states):
            j = table[q, c]
            for v in range(16):
                if (v >> (j & 3)) & 1:
                    nibble[c, j >> 2, v] |= 1 << q
    # only the few subsets reached are touched, and index is reset to -1 after
    index, subsets, successor = workspace
    index[final_mask] = 0
    subsets[0] = final_mask
    size = 1
    head = 0
    while head < size:
        for c in range(alphabet_size):
            image = 0
            for i in range(num_nibbles):
                image |= nibble[c, i, (subsets[head] >> (4 * i)) & 15]
            if index[image] < 0:
                index[image] = size
                subsets[size] = image
                size += 1
            successor[head, c] = index[image]
        head += 1
    index[subsets[:size]] = -1
    return successor[:size]


@njit(cache=True)
def _steps_to_first(successor):
    """Per node of the graph ``i -> successor[i, c]``, the fewest steps to node 0,
    or -1 if it never gets there."""
    size, alphabet_size = successor.shape
    # sources[start[j]:start[j + 1]]: the nodes with an edge into j
    in_degree = np.zeros(size + 1, np.int64)
    for i in range(size):
        for c in range(alphabet_size):
            in_degree[successor[i, c] + 1] += 1
    start = np.cumsum(in_degree)
    filled = np.zeros(size, np.int64)
    sources = np.empty(size * alphabet_size, np.int64)
    for i in range(size):
        for c in range(alphabet_size):
            j = successor[i, c]
            sources[start[j] + filled[j]] = i
            filled[j] += 1
    steps = np.full(size, -1, np.int64)
    steps[0] = 0
    queue = np.zeros(size, np.int64)
    queue_head, queue_end = 0, 1
    while queue_head < queue_end:
        j = queue[queue_head]
        queue_head += 1
        for i in sources[start[j] : start[j + 1]]:
            if steps[i] < 0:
                steps[i] = steps[j] + 1
                queue[queue_end] = i
                queue_end += 1
    return steps


@njit(cache=True)
def _class_preserving_fraction(table, final_mask, length, reject_below, workspace):
    """
    `preconditions.class_preserving_fraction`, or -1 once it is sure to be below
    ``reject_below``: mass pushed from F along `_subsets_reachable`, dropping
    what cannot get back to F in the steps left.
    """
    num_states, alphabet_size = table.shape
    if final_mask in (0, (1 << num_states) - 1):
        return 1.0
    successor = _subsets_reachable(table, final_mask, workspace)
    steps_to_final = _steps_to_first(successor)
    mass = np.zeros(len(successor))
    mass[0] = 1.0
    pushed = np.empty(len(successor))
    for t in range(length):
        steps_left = length - 1 - t
        pushed[:] = 0.0
        live = 0.0
        for i, share in enumerate(mass / alphabet_size):
            if share != 0.0:
                for j in successor[i]:
                    if 0 <= steps_to_final[j] <= steps_left:
                        pushed[j] += share
                        live += share
        # live mass bounds the answer from above
        if live < reject_below:
            return -1.0
        mass[:] = pushed
    return mass[0]


@njit(cache=True)
def _search(seed, table, length, min_fraction, max_candidates):
    """
    Draws candidates from numba's generator seeded with ``seed`` until one is
    minimal with class-preserving fraction at least ``min_fraction``, leaving
    its transitions in ``table`` and returning its accepting states as a mask;
    -1 if ``max_candidates`` run out.
    """
    np.random.seed(seed)
    num_states = table.shape[0]
    workspace = _workspace(num_states, table.shape[1])
    for _ in range(max_candidates):
        _draw_accessible(table)
        final_mask = np.random.randint(0, 1 << num_states)
        if final_mask in (0, (1 << num_states) - 1):
            continue
        if not _is_minimal(table, final_mask):
            continue
        fraction = _class_preserving_fraction(
            table, final_mask, length, min_fraction, workspace
        )
        if fraction >= 0:
            return final_mask
    return -1


def sample_minimal_admitted_dfa(
    rng: np.random.Generator, *, num_states: int, alphabet_size: int, length: int
) -> DFA:
    """
    A uniformly random transition table on ``num_states`` states in which every
    state is reachable from state 0, with each state accepting with probability
    1/2, kept if it is minimal and satisfies the preconditions at ``length``.
    """
    table = np.empty((num_states, alphabet_size), np.int64)
    while True:
        final_mask = _search(
            int(rng.integers(2**32)),
            table,
            length,
            preconditions.MIN_CLASS_PRESERVING_FRAC - _SLACK,
            _CANDIDATES_PER_SEARCH,
        )
        if final_mask < 0:
            continue
        aut = dfa_from_table(table, final_mask)
        if preconditions.satisfies_preconditions(aut, length=length):
            return aut
