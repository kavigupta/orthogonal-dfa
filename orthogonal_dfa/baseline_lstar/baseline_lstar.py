"""
Baseline L* implementation using aalpy.

Wraps our Oracle interface into aalpy's SUL interface and runs
standard L* to learn a DFA. This serves as a baseline comparison
against the orthonormal L* approach.
"""

from typing import List

from aalpy.base import SUL
from aalpy.oracles import RandomWordEqOracle
from automata.fa.dfa import DFA

from orthogonal_dfa.l_star.structures import Oracle

from .baseline_lstar_loop import run_lstar_with_max_states


class OracleSUL(SUL):
    """Wraps an Oracle into aalpy's SUL (System Under Learning) interface."""

    def __init__(self, oracle: Oracle):
        super().__init__()
        self.oracle = oracle
        self.current_word: List[int] = []

    def pre(self):
        self.current_word = []

    def post(self):
        pass

    def step(self, letter):
        if letter is not None:
            self.current_word.append(letter)
        return self.oracle.membership_query(list(self.current_word))


def run_baseline_lstar(oracle: Oracle, *, max_states=None):
    """
    Run standard L* on the given oracle and return an automata-lib DFA.

    Args:
        max_states: cap on number of states in the observation table's S set.
            When hit, the table is treated as closed and a hypothesis is built
            with unclosed rows mapped to the nearest S-row by suffix signature.
    """
    alphabet = list(range(oracle.alphabet_size))
    sul = OracleSUL(oracle)
    eq_oracle = RandomWordEqOracle(
        alphabet,
        sul,
        num_walks=5000,
        min_walk_len=5,
        max_walk_len=50,
    )

    learned = run_lstar_with_max_states(
        alphabet,
        sul,
        eq_oracle,
        max_states=max_states,
    )

    return aalpy_dfa_to_automata_lib(learned, alphabet)


def aalpy_dfa_to_automata_lib(aalpy_dfa, alphabet):
    """Convert an aalpy Dfa to an automata-lib DFA."""
    states = aalpy_dfa.states
    state_map = {s: i for i, s in enumerate(states)}

    transitions = {}
    for state in states:
        transitions[state_map[state]] = {}
        for sym in alphabet:
            next_state = state.transitions[sym]
            transitions[state_map[state]][sym] = state_map[next_state]

    initial_state = state_map[aalpy_dfa.initial_state]
    final_states = {state_map[s] for s in states if s.is_accepting}

    return DFA(
        states=set(range(len(states))),
        input_symbols=set(alphabet),
        transitions=transitions,
        initial_state=initial_state,
        final_states=final_states,
    )
