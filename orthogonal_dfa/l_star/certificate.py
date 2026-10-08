"""Whether a hypothesis's error against the noiseless target is certified.

For a DFA h, a state S of it reached by a share m_S of the sampler's strings,
and the oracle's read O of a string x drawn by the sampler, noise that depends
only on the label makes

    r_S = P(O = 1 | x reaches S) = p_0 + (p_1 - p_0) c_S,   c_S = P(f(x) = 1 | x reaches S),

with f the noiseless labels.  h's error is

    P(h(x) != f(x)) = sum_S m_S e_S,   e_S = c_S where S rejects, 1 - c_S where it accepts,

and p_0 is the only unknown besides the c_S: a state h labels without error
reads exactly at its label's rate, which pins it.
"""

import math
from typing import List, NamedTuple

import numpy as np
import scipy.stats


def look_level(alpha, look) -> float:
    """alpha 6 / (pi (look + 1))^2, which sums to alpha over every look."""
    return alpha * 6 / (math.pi * (look + 1)) ** 2


def error_bound(masses, rates, accepting, gap) -> float:
    """The largest sum_S m_S e_S, which worst_contributions spells out by state."""
    return float(worst_contributions(masses, rates, accepting, gap).sum())


def worst_contributions(masses, rates, accepting, gap) -> np.ndarray:
    """The m_S e_S of each state S at the masses, rates and offset that make

        sum_S m_S e_S

    largest over every m with sum_S m_S = 1 and m_S in
    [mass_low_S, mass_high_S], every r_S in [low_S, high_S], and every p_0 in
    [max_S low_S - band, min_S high_S], for masses = (mass_low, mass_high),
    rates = (low, high), c_S = (r_S - p_0) / band and

        band = max(gap, max_S low_S - min_S high_S),

    the narrowest band at least gap that the rates fit.  c_S is not held to
    [0, 1]: a state read past the band counts for more than its mass, as it does
    against E[O | h accepts] - E[O | h rejects].

    For a fixed p_0 each state's worst e_S is apart from the others', and the
    mass free of the lower bounds goes to the worst states first.  That is the
    largest of functions linear in p_0, so the largest over p_0 is at an end."""
    mass_low, mass_high, low, high = (
        np.asarray(v, dtype=float) for v in (*masses, *rates)
    )
    accepting = np.asarray(accepting, dtype=bool)
    band = max(gap, float(low.max() - high.min()))
    least = max(0.0, float(low.max()) - band)
    # Equal to least where band was set by the rates, up to rounding.
    most = max(least, min(1.0 - band, float(high.min())))

    def error(offset):
        share = (np.where(accepting, low, high) - offset) / band
        worst = np.where(accepting, 1 - share, share)
        weight = mass_low.copy()
        spare = 1 - weight.sum()
        for state in np.argsort(-worst, kind="stable"):
            added = min(spare, mass_high[state] - mass_low[state])
            weight[state] += added
            spare -= added
        return weight * worst

    return max((error(least), error(most)), key=lambda e: e.sum())


def clopper_pearson(hits, trials, level):
    """Per entry, the success probabilities (p_low, p_high) solving

        P(Binomial(trials, p_low) >= hits) = level / 2,
        P(Binomial(trials, p_high) <= hits) = level / 2,

    with p_low = 0 where hits = 0, p_high = 1 where hits = trials, and (0, 1)
    where trials = 0.  Whatever the success probability p a count is drawn at,
    P(p_low > p) <= level / 2 and P(p_high < p) <= level / 2."""
    low, high = np.zeros(len(trials)), np.ones(len(trials))
    seen = trials > 0
    hits, trials = hits[seen], trials[seen]
    low[seen] = np.where(
        hits > 0, scipy.stats.beta.ppf(level / 2, hits, trials - hits + 1), 0.0
    )
    high[seen] = np.where(
        hits < trials,
        scipy.stats.beta.ppf(1 - level / 2, hits + 1, trials - hits),
        1.0,
    )
    return low, high


class Verdict(NamedTuple):
    certified: bool
    #: The hypothesis the verdict is on.
    dfa: object
    #: The state with the largest m_S e_S at the rates read, and that e_S.
    blamed: object
    share: float


def certifies(pst, dfas, *, alpha) -> Verdict:
    """Whether any of ``dfas`` is certified, each tested on the same draws at
    ``alpha / len(dfas)``, so that

        P(some h certifies and sum_S m_S e_S > e) <= alpha,   e = certified_error,

    m_S the share of the sampler's strings reaching S, where the oracle's signal
    is min_signal_strength s exactly.  Whatever the signal, with m_A the share h
    accepts and m_R = 1 - m_A,

        P(h certifies, m_A m_R >= e and Delta < 2 s (1 - e / (m_A m_R))) <= alpha,
        Delta = E[O | h accepts] - E[O | h rejects]:

    h reads apart by as much as a DFA that errs on e of the strings would at
    signal s exactly.

    Look k reads n_k strings of the sampler, doubling, and certifies h when the
    error_bound at gap 2 s, over Clopper-Pearson intervals on each state's share
    of the draws and on its rate of O = 1, each at look_level(alpha, k) / 2K for
    K states, is at most e.  The verdict is on the first h certified, or, where
    every one is refused, on the one read with the least error."""
    tests = [_Test(dfa, pst.sampler.length) for dfa in dfas]
    gap = 2 * pst.config.min_signal_strength
    error = pst.config.certified_error
    size = max(len(test.states) for test in tests)
    drawn = 0
    refused = []
    look = 0
    while True:
        strings = [
            pst.sampler.sample(pst.rng, alphabet_size=pst.alphabet_size)
            for _ in range(size - drawn)
        ]
        drawn = size
        reads = pst.oracle.membership_queries(strings)
        level = look_level(alpha, look) / len(dfas)
        read = [
            test.read(strings, reads, size=size, level=level, gap=gap) for test in tests
        ]
        bound, at_rates, _, _ = min(read, key=lambda r: r[0])
        print(
            f"  certificate look {look}: {size} strings, error at most {bound:.4f}, "
            f"{at_rates:.4f} at the rates read, against {error}"
        )
        for test, (bound, at_rates, slack, verdict) in zip(tests, read):
            if bound <= error:
                return verdict._replace(certified=True)
            # Refusing carries no guarantee, only a round.  The slack is a rough
            # reach of the intervals, not a bound: refuse once the rates read are
            # further above e than it, or once it is so small that only a DFA
            # within e / 2 of the bar could still be undecided.
            if at_rates - error > slack or slack <= error / 2:
                refused.append((at_rates, verdict))
        refusing = {id(verdict.dfa) for _, verdict in refused}
        tests = [test for test in tests if id(test.dfa) not in refusing]
        if not tests:
            return min(refused, key=lambda r: r[0])[1]
        size *= 2
        look += 1


class _Test:
    """One hypothesis's counts over the certificate's draws."""

    def __init__(self, dfa, length):
        self.dfa = dfa
        self.states = sorted(dfa.states, key=str)
        self.accepting = np.array([state in dfa.final_states for state in self.states])
        self._route = _router(dfa, self.states, length)
        self._ones = np.zeros(len(self.states), dtype=int)
        self._drawn = np.zeros(len(self.states), dtype=int)

    def read(self, strings, reads, *, size, level, gap):
        """``(bound, at_rates, slack, verdict)`` once ``strings``, read as
        ``reads``, are counted too."""
        reached = self._route(strings)
        np.add.at(self._drawn, reached, 1)
        np.add.at(self._ones, reached, reads)
        drawn, ones = self._drawn, self._ones
        level /= 2 * len(self.states)
        masses = clopper_pearson(drawn, np.full(len(self.states), size), level)
        rates = clopper_pearson(ones, drawn, level)
        bound = error_bound(masses, rates, self.accepting, gap)
        shares, read = drawn / size, ones / np.maximum(drawn, 1)
        blame = worst_contributions(
            (shares, shares),
            (read, np.where(drawn > 0, read, 1.0)),
            self.accepting,
            gap,
        )
        worst = int(np.argmax(blame))
        share = float(blame[worst] / shares[worst]) if shares[worst] else 0.0
        slack = (masses[1] @ (rates[1] - rates[0])) / gap + np.sum(
            masses[1] - masses[0]
        )
        verdict = Verdict(False, self.dfa, self.states[worst], share)
        return bound, float(blame.sum()), slack, verdict


def _router(dfa, states, length):
    """A map from strings of the given length to the index in states of the
    state each reaches."""
    index = {state: i for i, state in enumerate(states)}
    step = np.zeros((len(states), max(dfa.input_symbols) + 1), dtype=np.int64)
    for state in states:
        assert len(dfa.transitions[state]) == len(dfa.input_symbols), state
        for symbol, target in dfa.transitions[state].items():
            step[index[state], symbol] = index[target]

    def route(strings: List[bytes]) -> np.ndarray:
        assert all(len(string) == length for string in strings), length
        read = np.frombuffer(b"".join(strings), dtype=np.uint8).reshape(-1, length)
        at = np.full(len(strings), index[dfa.initial_state], dtype=np.int64)
        for column in read.T:
            at = step[at, column]
        return at

    return route
