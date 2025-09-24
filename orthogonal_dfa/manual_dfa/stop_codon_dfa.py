from functools import lru_cache
import itertools

import pythomata


def evolve_state(state, symbol):
    done_mask, current_phase, codon_start = state

    done_mask = list(done_mask)
    current_phase = (current_phase + 1) % 3
    new_codon_start = codon_start + symbol

    if new_codon_start in ["TAA", "TAG", "TGA"]:
        done_mask[(current_phase + 1) % 3] = 1
        new_codon_start = ""
    while new_codon_start not in ["", "T", "TA", "TG"]:
        new_codon_start = new_codon_start[1:]

    done_mask = tuple(done_mask)
    new_state = done_mask, current_phase, new_codon_start
    return new_state


@lru_cache(None)
def stop_codon_dfa():
    states = {
        (done_mask, current_phase, codon_start)
        for done_mask in itertools.product([0, 1], repeat=3)
        for current_phase in range(3)
        for codon_start in ["", "T", "TA", "TG"]
    }
    alphabet = ["A", "C", "G", "T"]
    initial_state = ((0, 0, 0), 0, "")
    accepting_states = {state for state in states if state[0] == (1, 1, 1)}
    transitions = {}
    for state in states:
        transitions[state] = {}
        for symbol in alphabet:
            new_state = evolve_state(state, symbol)
            assert new_state in states, f"{new_state} not in {states}"
            transitions[state][symbol] = new_state
    dfa = pythomata.SimpleDFA(
        states=states,
        alphabet=alphabet,
        transition_function=transitions,
        initial_state=initial_state,
        accepting_states=accepting_states,
    )
    return dfa.minimize()
