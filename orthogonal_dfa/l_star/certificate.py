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
from typing import List

import numpy as np
import scipy.stats


def look_level(alpha, look) -> float:
    """alpha 6 / (pi (look + 1))^2, which sums to alpha over every look."""
    return alpha * 6 / (math.pi * (look + 1)) ** 2


def error_bound(masses, rates, accepting, gap) -> float:
    """The largest sum_S m_S e_S over every m with sum_S m_S = 1 and m_S in
    [mass_low_S, mass_high_S], every r_S in [low_S, high_S], and every p_0 with
    each c_S = (r_S - p_0) / band in [0, 1], for masses = (mass_low, mass_high)
    and rates = (low, high),

        band = max(gap, max_S low_S - min_S high_S),

    the narrowest band at least gap that the rates fit.

    For a fixed p_0 each state's worst e_S is apart from the others', and the
    mass free of the lower bounds goes to the worst states first.  For fixed m
    that is piecewise linear in p_0, kinked where a state's worst rate meets a
    bound, so the largest over p_0 is at those kinks or at the ends of the p_0
    that fit."""
    mass_low, mass_high, low, high = (
        np.asarray(v, dtype=float) for v in (*masses, *rates)
    )
    accepting = np.asarray(accepting, dtype=bool)
    band = max(gap, float(low.max() - high.min()))
    least = max(0.0, float(low.max()) - band)
    # Equal to least where band was set by the rates, up to rounding.
    most = max(least, min(1.0 - band, float(high.min())))

    def error(offset):
        rate = np.where(
            accepting, np.maximum(low, offset), np.minimum(high, offset + band)
        )
        share = (rate - offset) / band
        worst = np.where(accepting, 1 - share, share)
        weight = mass_low.copy()
        spare = 1 - weight.sum()
        for state in np.argsort(-worst, kind="stable"):
            added = min(spare, mass_high[state] - mass_low[state])
            weight[state] += added
            spare -= added
        return float(weight @ worst)

    corners = np.concatenate([[least, most], high - band, low])
    return max(error(p) for p in corners if least <= p <= most)


def _intervals(hits, trials, level):
    """Per entry, for hits ~ Binomial(trials, p), the Clopper-Pearson bounds with

        P(low > p) <= level / 2   and   P(high < p) <= level / 2

    whatever p is; (0, 1) where trials is 0."""
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


def certifies(pst, dfa, *, alpha) -> bool:
    """True only with

        P(certifies and sum_S m_S e_S > e) <= alpha,   e = certified_error,

    m_S the share of the sampler's strings reaching S, where the oracle's signal
    is min_signal_strength s exactly; nothing is guaranteed where p_1 - p_0 > 2 s.

    Look k reads n_k strings of the sampler, doubling, and certifies when the
    error_bound at gap 2 s, over Clopper-Pearson intervals on each state's share
    of the draws and on its rate of O = 1, each at look_level(alpha, k) / 2K for
    K states, is at most e."""
    states = sorted(dfa.states, key=str)
    accepting = np.array([state in dfa.final_states for state in states])
    route = _router(dfa, states, pst.sampler.length)
    gap = 2 * pst.config.min_signal_strength
    error = pst.config.certified_error
    ones = np.zeros(len(states), dtype=int)
    drawn = np.zeros(len(states), dtype=int)
    size = len(states)
    look = 0
    while True:
        strings = [
            pst.sampler.sample(pst.rng, alphabet_size=pst.alphabet_size)
            for _ in range(size - drawn.sum())
        ]
        reached = route(strings)
        np.add.at(drawn, reached, 1)
        np.add.at(ones, reached, pst.oracle.membership_queries(strings))
        level = look_level(alpha, look) / (2 * len(states))
        masses = _intervals(drawn, np.full(len(states), size), level)
        rates = _intervals(ones, drawn, level)
        bound = error_bound(masses, rates, accepting, gap)
        shares, read = drawn / size, ones / np.maximum(drawn, 1)
        at_rates = error_bound(
            (shares, shares), (read, np.where(drawn > 0, read, 1.0)), accepting, gap
        )
        slack = (masses[1] @ (rates[1] - rates[0])) / gap + np.sum(
            masses[1] - masses[0]
        )
        print(
            f"  certificate look {look}: {size} strings, error at most {bound:.4f}, "
            f"{at_rates:.4f} at the rates read, against {error}"
        )
        if bound <= error:
            return True
        # Refusing carries no guarantee, only a round.  The slack is a rough
        # reach of the intervals, not a bound: refuse once the rates read are
        # further above e than it, or once it is so small that only a DFA within
        # e / 2 of the bar could still be undecided.
        if at_rates - error > slack or slack <= error / 2:
            return False
        size *= 2
        look += 1


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
