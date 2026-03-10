def states_intermediate(s0, y, dfa):
    states = [s0]
    for symbol in y:
        s_next = dfa.transitions[states[-1]][symbol]
        states.append(s_next)
    return states
