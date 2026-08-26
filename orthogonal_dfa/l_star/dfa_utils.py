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
        row = counts[remaining - 1]
        transitions = dfa.transitions[state]
        weights = [row[transitions[s]] for s in syms]
        # One uniform draw walked against the weights rather than rng.choice(p=),
        # which spends ~20us normalising and building a CDF for what is usually a
        # two-way split, once per symbol of every sampled string.
        target = rng.random() * sum(weights)
        cumulative = 0
        symbol = syms[0]
        for candidate, weight in zip(syms, weights):
            if not weight:
                continue
            # Never leaves a zero-weight symbol selected, and a draw that overruns
            # the total on rounding lands on the last symbol that had mass.
            symbol = candidate
            cumulative += weight
            if target < cumulative:
                break
        string.append(symbol)
        state = transitions[symbol]
    return string


def rank_string_reaching_state(dfa, counts, string):
    """
    Index in ``[0, counts[length][initial])`` of ``string`` among the length-``length``
    strings reaching the target, in the order used by :func:`unrank_string_reaching_state`.
    """
    syms = sorted(dfa.input_symbols)
    length = len(counts) - 1
    state = dfa.initial_state
    index = 0
    for i, symbol in enumerate(string):
        remaining = length - i
        for s in syms:
            if s == symbol:
                break
            index += counts[remaining - 1][dfa.transitions[state][s]]
        state = dfa.transitions[state][symbol]
    return index


def unrank_string_reaching_state(dfa, counts, index):
    """Inverse of :func:`rank_string_reaching_state`: the ``index``-th length-``length``
    string reaching the target ``counts`` was built for.
    """
    syms = sorted(dfa.input_symbols)
    length = len(counts) - 1
    state = dfa.initial_state
    string = []
    for remaining in range(length, 0, -1):
        for s in syms:
            block = counts[remaining - 1][dfa.transitions[state][s]]
            if index < block:
                string.append(s)
                state = dfa.transitions[state][s]
                break
            index -= block
    return string


def per_state_sample(dfa, rng, length, per_state, existing=()):
    """
    Sample strings of length ``length`` such that when these strings are
    added to the list of strings already in ``existing``, each state of ``dfa``
    has at least ``per_state`` strings.
    """
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
        used = set()
        for s in have.get(state, ()):
            if len(used) >= target:
                break
            used.add(rank_string_reaching_state(dfa, counts, s))
        fits_int64 = reachable <= 2**63 - 1
        while len(used) < target:
            if fits_int64:
                used.add(int(rng.integers(reachable)))
            else:
                used.add(
                    rank_string_reaching_state(
                        dfa, counts, sample_string_reaching_state(dfa, counts, rng)
                    )
                )
        pool.extend(unrank_string_reaching_state(dfa, counts, r) for r in used)
    return pool
