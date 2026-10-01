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

from .dfa_utils import count_paths_to_state


def look_level(alpha, look) -> float:
    """alpha 6 / (pi (look + 1))^2, which sums to alpha over every look."""
    return alpha * 6 / (math.pi * (look + 1)) ** 2


def error_bound(masses, accepting, low, high, gap) -> float:
    """The largest sum_S m_S e_S, m_S = masses[S], over every r_S in
    [low_S, high_S] and every p_0 with each c_S = (r_S - p_0) / band in [0, 1],

        band = max(gap, max_S low_S - min_S high_S),

    the narrowest band at least gap that the rates fit."""
    return _largest_error(masses, accepting, low, high, gap, worst=True)


def error_floor(masses, accepting, low, high, gap) -> float:
    """As error_bound, but with each r_S the one in [low_S, high_S] that makes
    the error least for the p_0: no rates in the intervals have an error_bound
    below it."""
    return _largest_error(masses, accepting, low, high, gap, worst=False)


def _largest_error(masses, accepting, low, high, gap, *, worst):
    """For a fixed p_0 each state's rate is chosen apart from the others', so the
    error is piecewise linear in p_0, and the largest is at the ends of the p_0
    that fit or where a state's chosen rate meets a bound."""
    masses, low, high = (np.asarray(v, dtype=float) for v in (masses, low, high))
    accepting = np.asarray(accepting, dtype=bool)
    band = max(gap, float(low.max() - high.min()))
    least = max(0.0, float(low.max()) - band)
    # Equal to least where band was set by the rates, up to rounding.
    most = max(least, min(1.0 - band, float(high.min())))
    raising = accepting != worst

    def error(offset):
        rate = np.where(
            raising, np.minimum(high, offset + band), np.maximum(low, offset)
        )
        share = (rate - offset) / band
        return float(masses @ np.where(accepting, 1 - share, share))

    corners = np.concatenate([[least, most], high - band, low])
    return max(error(p) for p in corners if least <= p <= most)


def _rate_intervals(ones, drawn, level):
    """Clopper-Pearson bounds on each state's rate of O = 1, each at level / 2
    a side; [0, 1] for a state with no draws."""
    low = np.zeros(len(drawn))
    high = np.ones(len(drawn))
    seen = drawn > 0
    hits, trials = ones[seen], drawn[seen]
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

    where the oracle's signal is min_signal_strength s exactly.

    Look k reads n_k strings of the sampler, doubling, and certifies when the
    error_bound at gap 2 s over every state's Clopper-Pearson interval, each
    at look_level(alpha, k) over the states drawn into, is at most e, and
    refuses once the error_floor over those intervals is above e."""
    states, masses = _state_masses(pst, dfa)
    accepting = [state in dfa.final_states for state in states]
    route = _router(dfa, states)
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
        assert (reached >= 0).all(), "a string reached a state the sampler cannot"
        read = np.asarray(pst.oracle.membership_queries(strings), dtype=int)
        np.add.at(drawn, reached, 1)
        np.add.at(ones, reached, read)
        level = look_level(alpha, look) / max(1, int((drawn > 0).sum()))
        low, high = _rate_intervals(ones, drawn, level)
        bound = error_bound(masses, accepting, low, high, gap)
        floor = error_floor(masses, accepting, low, high, gap)
        print(
            f"  certificate look {look}: {size} strings, error at most {bound:.4f} "
            f"and, at the most favourable rates, at least {floor:.4f}, against {error}"
        )
        if bound <= error:
            return True
        # Refusing carries no guarantee, only a round.
        if floor > error:
            return False
        size *= 2
        look += 1


def _state_masses(pst, dfa):
    """(states, m): the states the sampler reaches, and the share of its
    strings reaching each, its positions drawn independently by its symbol
    weights."""
    weights = pst.sampler.symbol_weights(pst.alphabet_size)
    length = pst.sampler.length
    states = sorted(dfa.states, key=str)
    mass = np.array(
        [
            float(
                count_paths_to_state(dfa, state, length, weights)[length][
                    dfa.initial_state
                ]
            )
            for state in states
        ]
    )
    reached = mass > 0
    return [s for s, r in zip(states, reached) if r], mass[reached] / mass.sum()


def _router(dfa, states):
    """A function from equal-length strings to the index in states of the state
    each reaches."""
    every = sorted(dfa.states, key=str)
    index = {state: i for i, state in enumerate(every)}
    step = np.zeros((len(every), max(dfa.input_symbols) + 1), dtype=np.int64)
    for state in every:
        for symbol, target in dfa.transitions[state].items():
            step[index[state], symbol] = index[target]
    position = np.full(len(every), -1, dtype=np.int64)
    position[[index[state] for state in states]] = np.arange(len(states))

    def route(strings: List[bytes]) -> np.ndarray:
        read = np.frombuffer(b"".join(strings), dtype=np.uint8).reshape(
            len(strings), -1
        )
        at = np.full(len(strings), index[dfa.initial_state], dtype=np.int64)
        for column in read.T:
            at = step[at, column]
        return position[at]

    return route
