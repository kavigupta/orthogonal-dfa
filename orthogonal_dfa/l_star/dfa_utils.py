import numpy as np


def states_intermediate(s0, y, dfa):
    states = [s0]
    for symbol in y:
        s_next = dfa.transitions[states[-1]][symbol]
        states.append(s_next)
    return states


def count_paths_to_state(dfa, target, length):
    """``counts[m][q]`` = number of length-``m`` strings ``w`` with ``run(q, w) == target``.

    Standard path-counting DP, for ``m`` in ``0..length``: enough to sample a uniform
    length-``length`` string reaching ``target`` via :func:`sample_string_reaching_state`.
    """
    syms = sorted(dfa.input_symbols)
    counts = [{q: int(q == target) for q in dfa.states}]
    for _ in range(length):
        prev = counts[-1]
        counts.append(
            {q: sum(prev[dfa.transitions[q][s]] for s in syms) for q in dfa.states}
        )
    return counts


def sample_string_reaching_state(dfa, counts, rng):
    """Uniform random length-``len(counts)-1`` string from ``dfa.initial_state`` to the
    target ``counts`` was built for, or ``None`` if no such string exists.

    The recursive sampling method: walk forward choosing each symbol with probability
    proportional to the number of completions that still reach the target.
    """
    syms = sorted(dfa.input_symbols)
    length = len(counts) - 1
    state = dfa.initial_state
    if counts[length][state] == 0:
        return None
    string = []
    for remaining in range(length, 0, -1):
        weights = np.array(
            [counts[remaining - 1][dfa.transitions[state][s]] for s in syms], float
        )
        symbol = syms[rng.choice(len(syms), p=weights / weights.sum())]
        string.append(symbol)
        state = dfa.transitions[state][symbol]
    return string


def per_state_sample(dfa, rng, length, per_state, existing=()):
    """A state-balanced selection: ``per_state`` distinct length-``length`` strings
    reaching *each* state of ``dfa``, drawn with the path-counting sampler.

    Length-``length`` strings in ``existing`` that already reach a state count
    toward its ``per_state`` target and are reused as its representatives, so
    repeated calls top the coverage up to ``per_state`` rather than adding a fresh
    ``per_state`` every time -- the pool converges instead of building up. The
    returned selection includes both reused and freshly-drawn strings.

    Keep in sync with ``lstar.denoise_accept_labels.relabel``, which runs the same
    per-state count-paths-then-draw-distinct loop (it also scores accepts)."""
    have = {}
    for s in existing:
        if len(s) == length:
            have.setdefault(
                states_intermediate(dfa.initial_state, s, dfa)[-1], []
            ).append(tuple(s))
    pool = []
    for state in sorted(dfa.states):
        counts = count_paths_to_state(dfa, state, length)
        reachable = counts[length][dfa.initial_state]
        if reachable == 0:
            continue
        target = min(per_state, reachable)
        seen = set(have.get(state, ())[:target])
        for _ in range(per_state * 5):
            if len(seen) >= target:
                break
            seen.add(tuple(sample_string_reaching_state(dfa, counts, rng)))
        pool.extend(list(s) for s in seen)
    return pool
