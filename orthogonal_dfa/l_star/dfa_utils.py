import numpy as np


def states_intermediate(s0, y, dfa):
    states = [s0]
    for symbol in y:
        s_next = dfa.transitions[states[-1]][symbol]
        states.append(s_next)
    return states


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
