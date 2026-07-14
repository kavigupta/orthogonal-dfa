import numpy as np


def states_intermediate(s0, y, dfa):
    states = [s0]
    for symbol in y:
        s_next = dfa.transitions[states[-1]][symbol]
        states.append(s_next)
    return states


def count_paths_to_state(dfa, target, length):
    """``counts[m][q]`` = number of length-``m`` strings ``w`` with ``run(q, w) == target``.

    Standard path-counting DP, for ``m`` in ``0..length``: enough to sample a uniform
    length-``length`` string reaching ``target`` via :func:`sample_string_reaching_state`.
    """
    syms = sorted(dfa.input_symbols)
    counts = [{q: int(q == target) for q in dfa.states}]
    for _ in range(length):
        prev = counts[-1]
        counts.append(
            {q: sum(prev[dfa.transitions[q][s]] for s in syms) for q in dfa.states}
        )
    return counts


def sample_string_reaching_state(dfa, counts, rng):
    """Uniform random length-``len(counts)-1`` string from ``dfa.initial_state`` to the
    target ``counts`` was built for, or ``None`` if no such string exists.

    The recursive sampling method: walk forward choosing each symbol with probability
    proportional to the number of completions that still reach the target.
    """
    syms = sorted(dfa.input_symbols)
    length = len(counts) - 1
    state = dfa.initial_state
    if counts[length][state] == 0:
        return None
    string = []
    for remaining in range(length, 0, -1):
        weights = np.array(
            [counts[remaining - 1][dfa.transitions[state][s]] for s in syms], float
        )
        symbol = syms[rng.choice(len(syms), p=weights / weights.sum())]
        string.append(symbol)
        state = dfa.transitions[state][symbol]
    return string


def final_states_all_initial(transitions, strings):
    """For each possible initial state, compute the final state of each string.

    Returns an array of shape (num_states, len(strings)) where
    result[initial_state, string_idx] is the final state reached.
    """
    num_states = transitions.shape[0]
    result = np.tile(np.arange(num_states)[:, None], (1, len(strings)))
    for string_idx, string in enumerate(strings):
        for sym in string:
            result[:, string_idx] = transitions[result[:, string_idx], sym]
    return result
