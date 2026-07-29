"""Small hand-written DFAs shared by the unit tests."""

from automata.fa.dfa import DFA

# Accepts strings with an even number of 1s.  State 0 is the accepting one.
PARITY = DFA(
    states={0, 1},
    input_symbols={0, 1},
    transitions={0: {0: 0, 1: 1}, 1: {0: 1, 1: 0}},
    initial_state=0,
    final_states={0},
    allow_partial=False,
)


def parity_membership(string) -> bool:
    """The language PARITY recognises, for a noiseless test oracle."""
    return sum(string) % 2 == 0
