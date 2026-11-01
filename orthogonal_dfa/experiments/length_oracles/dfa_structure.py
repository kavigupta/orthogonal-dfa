"""Structure of a DFA over the length-oracle vocabulary: symbols 0-2 are stops, 3 and 4
wildcards."""


def is_ring(dfa):
    """A single cycle of at least 3 states, every other edge a self-loop: a counter."""
    successors = {
        q: {t for t in dfa.transitions[q].values() if t != q} for q in dfa.states
    }
    if len(dfa.states) < 3 or any(len(t) != 1 for t in successors.values()):
        return False
    q, seen = dfa.initial_state, set()
    while q not in seen:
        seen.add(q)
        (q,) = successors[q]
    return q == dfa.initial_state and seen == set(dfa.states)


def counts_wildcards(dfa):
    """Some cycle of wildcard transitions passes both accepting and rejecting states, so
    the verdict moves with the wildcard count; a DFA tracking the phase instead has
    wildcard cycles of one verdict, and uses the phase only to send stops elsewhere."""
    if any(dfa.transitions[q][3] != dfa.transitions[q][4] for q in dfa.states):
        raise ValueError("the two wildcards go to different states")
    for q in dfa.states:
        path = []
        while q not in path:
            path.append(q)
            q = dfa.transitions[q][3]
        cycle = set(path[path.index(q) :])
        if 0 < len(cycle & dfa.final_states) < len(cycle):
            return True
    return False


def dfa_accepts(dfa, string):
    q = dfa.initial_state
    for c in string:
        q = dfa.transitions[q][c]
    return q in dfa.final_states
