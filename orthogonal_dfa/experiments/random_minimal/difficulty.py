"""Properties of a target that might make it hard for E-L*, all exact."""

from __future__ import annotations

from typing import Dict

import numpy as np
from automata.fa.dfa import DFA

from orthogonal_dfa.l_star import preconditions


def _table(dfa: DFA):
    states = sorted(dfa.states, key=str)
    index = {q: i for i, q in enumerate(states)}
    symbols = sorted(dfa.input_symbols)
    table = np.array([[index[dfa.transitions[q][c]] for c in symbols] for q in states])
    final = np.array([q in dfa.final_states for q in states])
    return table, final, index[dfa.initial_state]


def endpoint_distribution(table: np.ndarray, initial: int, length: int) -> np.ndarray:
    """Per state, the probability a random length-``length`` string ends there."""
    num_states, alphabet_size = table.shape
    step = np.zeros((num_states, num_states))
    for c in range(alphabet_size):
        np.add.at(step, (np.arange(num_states), table[:, c]), 1 / alphabet_size)
    return np.linalg.matrix_power(step, length)[initial]


def depths(table: np.ndarray, initial: int) -> np.ndarray:
    """Per state, the length of the shortest string reaching it."""
    depth = np.full(len(table), -1)
    depth[initial] = 0
    frontier = [initial]
    while frontier:
        reached = []
        for q in frontier:
            for r in table[q]:
                if depth[r] < 0:
                    depth[r] = depth[q] + 1
                    reached.append(r)
        frontier = reached
    return depth


def separating_lengths(table: np.ndarray, final: np.ndarray) -> np.ndarray:
    """``[p, q]``: the length of the shortest suffix telling p from q, -1 if none."""
    num_states = len(table)
    length = np.where(final[:, None] != final[None, :], 0, -1)
    for k in range(1, num_states):
        apart = length >= 0
        split = np.zeros_like(apart)
        for successors in table.T:
            split |= apart[np.ix_(successors, successors)]
        length[split & (length < 0)] = k
    return length


def synchronizing_probability(table: np.ndarray, length: int) -> float:
    """The probability a random length-``length`` string sends every state to
    one state."""
    num_states, alphabet_size = table.shape
    images = {(1 << num_states) - 1: 1.0}
    for _ in range(length):
        nxt: Dict[int, float] = {}
        for subset, mass in images.items():
            for c in range(alphabet_size):
                image = 0
                for q in range(num_states):
                    if (subset >> q) & 1:
                        image |= 1 << int(table[q, c])
                nxt[image] = nxt.get(image, 0.0) + mass / alphabet_size
        images = nxt
    return sum(mass for subset, mass in images.items() if subset & (subset - 1) == 0)


def target_metrics(dfa: DFA, *, length: int) -> Dict:
    report = preconditions.satisfies_preconditions(
        dfa, length=length, short_circuit=False
    )
    table, final, initial = _table(dfa)
    endpoints = endpoint_distribution(table, initial, length)
    separating = separating_lengths(table, final)
    off_diagonal = separating[~np.eye(len(table), dtype=bool)]
    return {
        "states": len(table),
        "alphabet_size": table.shape[1],
        "accepting_states": int(final.sum()),
        "acceptance_rate": report.acceptance_rate,
        "class_preserving_fraction": report.class_preserving_fraction,
        "non_start_zone_accuracy_ceiling": report.non_start_zone_accuracy_ceiling,
        "unvisited_states": len(report.unvisited_states),
        "starts": len(preconditions.non_start_zone_states(dfa, length=length)),
        "endpoint_mass_min": float(endpoints.min()),
        "endpoint_mass_max": float(endpoints.max()),
        "depth_max": int(depths(table, initial).max()),
        "separating_length_max": int(off_diagonal.max()),
        "separating_length_mean": float(off_diagonal.mean()),
        "synchronizing_probability": synchronizing_probability(table, length),
    }
