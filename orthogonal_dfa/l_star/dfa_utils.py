from automata.fa.dfa import DFA


def states_intermediate(s0, y, dfa):
    states = [s0]
    for symbol in y:
        s_next = dfa.transitions[states[-1]][symbol]
        states.append(s_next)
    return states


def random_word(dfa, size, rng):
    def to_str(digit):
        assert 0 <= digit <= 9
        return str(digit)

    dfa = DFA(
        states=dfa.states,
        initial_state=dfa.initial_state,
        input_symbols={to_str(x) for x in dfa.input_symbols},
        transitions={
            s1: {to_str(sym): s2 for sym, s2 in trans.items()}
            for s1, trans in dfa.transitions.items()
        },
        final_states=dfa.final_states,
    )
    return [int(x) for x in dfa.random_word(size, seed=int(rng.integers(0, 2**32 - 1)))]
