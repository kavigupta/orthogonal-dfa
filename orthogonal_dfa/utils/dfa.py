import pythomata
from permacache import stable_hash


def rename_symbols(dfa: pythomata.SimpleDFA, mapping: dict) -> pythomata.SimpleDFA:
    assert set(dfa.alphabet) == set(
        mapping.keys()
    ), f"DFA alphabet is {set(dfa.alphabet)}; mapping keys are {set(mapping.keys())}"
    new_alphabet = set(mapping.values())
    return pythomata.SimpleDFA(
        states=dfa.states,
        alphabet=new_alphabet,
        transition_function={
            state: {
                mapping[symbol]: new_state for symbol, new_state in transitions.items()
            }
            for state, transitions in dfa.transition_function.items()
        },
        initial_state=dfa.initial_state,
        accepting_states=dfa.accepting_states,
    )


def acgt_to_num(dfa: pythomata.SimpleDFA) -> pythomata.SimpleDFA:
    return rename_symbols(dfa, {"A": 0, "C": 1, "G": 2, "T": 3})


def hash_dfa(dfa: pythomata.SimpleDFA) -> str:
    return str(
        stable_hash(
            (
                sorted(dfa.states),
                sorted(dfa.alphabet),
                sorted(
                    (state, symbol, new_state)
                    for state, transitions in dfa.transition_function.items()
                    for symbol, new_state in transitions.items()
                ),
                dfa.initial_state,
                sorted(dfa.accepting_states),
            )
        )
    )
