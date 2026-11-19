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


def wildcard_cycles(dfa):
    """Every simple cycle that reads only wildcards, each as a set of states; where the
    two wildcards go to different states, a cycle may take either."""
    successors = {q: {dfa.transitions[q][3], dfa.transitions[q][4]} for q in dfa.states}
    cycles = set()

    def extend(path):
        for t in successors[path[-1]]:
            if t == path[0]:
                cycles.add(frozenset(path))
            elif t not in path:
                extend(path + [t])

    for q in dfa.states:
        extend([q])
    return cycles


def counts_wildcards(dfa):
    """Some wildcard cycle passes both accepting and rejecting states, so the verdict
    moves with the wildcard count."""
    return any(
        0 < len(cycle & dfa.final_states) < len(cycle) for cycle in wildcard_cycles(dfa)
    )


def tracks_phase(dfa):
    """Some wildcard cycle of a multiple of 3 states, all of one verdict, follows the
    reading frame (a stop is 3 bases, a wildcard 1), and a stop at one of its phases goes
    straight to a rejecting trap."""
    traps = {
        q
        for q in dfa.states - dfa.final_states
        if all(t == q for t in dfa.transitions[q].values())
    }
    return any(
        len(cycle) % 3 == 0
        and len(cycle & dfa.final_states) in (0, len(cycle))
        and any(dfa.transitions[q][stop] in traps for q in cycle for stop in (0, 1, 2))
        for cycle in wildcard_cycles(dfa)
    )


def dfa_accepts(dfa, string):
    q = dfa.initial_state
    for c in string:
        q = dfa.transitions[q][c]
    return q in dfa.final_states


def frame_states(dfa):
    """Each state's (phase, closed frames) over the strings that reach it.  The phase is
    the count of wildcards mod 3, and a stop closes the frame of the phase it is read
    at."""
    start = (0, frozenset())
    found = {q: set() for q in dfa.states}
    found[dfa.initial_state].add(start)
    frontier = [(dfa.initial_state, start)]
    while frontier:
        q, (phase, closed) = frontier.pop()
        for c, t in dfa.transitions[q].items():
            after = ((phase + 1) % 3, closed) if c >= 3 else (phase, closed | {phase})
            if after not in found[t]:
                found[t].add(after)
                frontier.append((t, after))
    return found
