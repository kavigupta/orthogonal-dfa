import itertools
from functools import lru_cache

import pythomata

from orthogonal_dfa.psams.psam_pdfa import PSAMPDFA
from orthogonal_dfa.psams.psams import TorchPSAMs
from orthogonal_dfa.utils.dfa import canonicalize_states
from orthogonal_dfa.utils.pdfa import PDFA
from orthogonal_dfa.utils.probability import ZeroProbability


def mark_phase(done_mask, current_phase, *, phase_agnostic):
    if sum(done_mask) == 3:
        return
    if phase_agnostic:
        done_mask.append(1)
    else:
        done_mask[(current_phase + 1) % 3] = 1


def evolve_state(state, symbol, stops, prefixes, *, phase_agnostic):
    done_mask, current_phase, codon_start = state

    done_mask = list(done_mask)
    current_phase = (current_phase + 1) % 3
    new_codon_start = codon_start + symbol

    if new_codon_start in stops:
        mark_phase(done_mask, current_phase, phase_agnostic=phase_agnostic)
        new_codon_start = ""
    while new_codon_start not in prefixes:
        new_codon_start = new_codon_start[1:]

    done_mask = tuple(done_mask)
    new_state = done_mask, current_phase, new_codon_start
    return new_state


@lru_cache(None)
def stop_codon_dfa(stops=("TAG", "TAA", "TGA"), *, phase_agnostic=False):
    prefixes = sorted(set(s[:i] for s in stops for i in range(len(s))))
    states = {
        (done_mask, current_phase, codon_start)
        for done_mask in (
            [(1,) * k for k in range(4)]
            if phase_agnostic
            else list(itertools.product([0, 1], repeat=3))
        )
        for current_phase in range(3)
        for codon_start in prefixes
    }
    alphabet = ["A", "C", "G", "T"]
    initial_state = (() if phase_agnostic else (0, 0, 0), 0, "")
    accepting_states = {state for state in states if state[0] == (1, 1, 1)}
    transitions = {}
    for state in states:
        transitions[state] = {}
        for symbol in alphabet:
            new_state = evolve_state(
                state, symbol, stops, prefixes, phase_agnostic=phase_agnostic
            )
            assert new_state in states, f"{new_state} not in {states}"
            transitions[state][symbol] = new_state
    dfa = pythomata.SimpleDFA(
        states=states,
        alphabet=alphabet,
        transition_function=transitions,
        initial_state=initial_state,
        accepting_states=accepting_states,
    )
    return canonicalize_states(dfa.minimize())


def stop_codon_from_psams(num_stops, phase_agnostic=False):
    start_state = ((0, 0, 0) if not phase_agnostic else (), 0)
    states = set()
    fringe = [start_state]
    accepting_states = {((1, 1, 1), 0), ((1, 1, 1), 1), ((1, 1, 1), 2)}
    transitions = {}
    while fringe:
        state = fringe.pop()
        if state in states:
            continue
        states.add(state)
        transitions[state] = {}
        done_mask, phase = state
        for symbol in range(1 + num_stops):
            new_phase = (phase + 1) % 3
            new_done_mask = list(done_mask)
            if symbol != 0:
                mark_phase(new_done_mask, phase, phase_agnostic=phase_agnostic)
            new_state = (tuple(new_done_mask), new_phase)
            transitions[state][symbol] = new_state
            fringe.append(new_state)
    dfa = pythomata.SimpleDFA(
        states=states,
        alphabet=list(range(1 + num_stops)),
        transition_function=transitions,
        initial_state=start_state,
        accepting_states=accepting_states,
    )
    return canonicalize_states(dfa.minimize())


def stop_codon_psamdfa(*stops, zero_prob: ZeroProbability, phase_agnostic=False):
    model = PSAMPDFA(
        TorchPSAMs.from_literal_strings(*stops, zero_prob=zero_prob),
        PDFA.from_dfa(
            stop_codon_from_psams(len(stops), phase_agnostic=phase_agnostic),
            zero_prob=zero_prob,
        ),
    )
    return model
