"""Whether a hypothesis's error against the noiseless target is certified.

For a DFA h and the oracle's read O of a string x drawn by the sampler,

    A(h) = P(O = 1 | h(x) = 1) - P(O = 1 | h(x) = 0).

Noise that depends only on the label makes A(h) = (p_1 - p_0) A_f(h), with A_f
the same difference against the noiseless labels f, and

    P(h(x) != f(x)) <= max(q, 1 - q) (1 - A_f(h)),   q = P(h(x) = 1),

so where p_1 - p_0 = 2 s, A(h) >= 2 s (1 - e / max(q, 1 - q)) holds h's error
within e.

The strings of each side are drawn as the state-reaching walks draw them, each
position independently by the sampler's symbol weights.
"""

import math
from typing import Tuple

import numpy as np
import scipy.stats

from .dfa_utils import count_paths_to_state, sample_string_reaching_state


class TooMuchRead(Exception):
    """More of the sampler's strings on the two sides of a DFA were read before
    than the certificate's slack allows it to leave unconstrained."""


def advantage_bounds(sides, level) -> Tuple[float, float]:
    """(lo, hi) with P(A < lo) <= level and P(A > hi) <= level, for

        A = P(O = 1 | h = 1) - P(O = 1 | h = 0)

    and sides[s] the (drawn, read, ones) of _rate_bounds given h = s."""
    low_1, high_1 = _rate_bounds(*sides[0], level / 4)
    low_0, high_0 = _rate_bounds(*sides[1], level / 4)
    return low_1 - high_0, high_1 - low_0


def _rate_bounds(drawn, read, ones, level) -> Tuple[float, float]:
    """(low, high) with P(r < low) <= 2 level and P(r > high) <= 2 level, for r
    the rate of O = 1 over what the drawn strings are drawn from: read of them
    were read before, and may read anything, and ones of the rest read 1."""
    _, read_high = _clopper_pearson(read, drawn, level)
    unread_low, unread_high = _clopper_pearson(ones, drawn - read, level)
    return (1 - read_high) * unread_low, unread_high + read_high * (1 - unread_high)


def _clopper_pearson(hits, drawn, level) -> Tuple[float, float]:
    low = scipy.stats.beta.ppf(level, hits, drawn - hits + 1) if hits else 0.0
    high = (
        scipy.stats.beta.ppf(1 - level, hits + 1, drawn - hits) if hits < drawn else 1.0
    )
    return float(low), float(high)


def look_level(alpha, look) -> float:
    """alpha 6 / (pi (look + 1))^2, which sums to alpha over every look."""
    return alpha * 6 / (math.pi * (look + 1)) ** 2


def first_look(target, level) -> int:
    """The least m with 2 (level / 4)^(2 / m) - 1 >= target: below it not even m
    strings a side, none read before and every one read as h reads it, clear the
    target at level."""
    return math.ceil(2 * math.log(level / 4) / math.log((1 + target) / 2))


def certifies(pst, dfa, *, alpha) -> bool:
    """Whether advantage_bounds, on m_k strings a side at look k drawn given
    dfa's label, at level look_level(alpha, k), put

        A(dfa) >= 2 s - d,   d = 2 s e / max(q, 1 - q),

    s = min_signal_strength, e = certified_error, q = P(dfa(x) = 1), before they
    put it below or narrow to d apart; m_k doubles from first_look at look 0's
    level.  P(certifies and A(dfa) < 2 s - d) <= alpha.  Raises TooMuchRead once
    more of the two sides was read before than d, which no further draw can
    undo."""
    (draw_1, mass_1), (draw_0, mass_0) = (
        _given_label(pst, dfa, True),
        _given_label(pst, dfa, False),
    )
    if not mass_1 or not mass_0:
        return False
    sides = [draw_1, draw_0]
    signal = pst.config.min_signal_strength
    accepted = mass_1 / (mass_1 + mass_0)
    slack = 2 * signal * pst.config.certified_error / max(accepted, 1 - accepted)
    target = 2 * signal - slack
    memo = pst.table.memo
    counts = np.zeros((2, 3), dtype=int)
    # Whether each string drawn was read before this certificate first drew it.
    read_before = {}
    size = first_look(target, look_level(alpha, 0))
    look = 0
    while True:
        level = look_level(alpha, look)
        for side, draw in enumerate(sides):
            strings = [draw() for _ in range(size - counts[side, 0])]
            for string in strings:
                if string not in read_before:
                    read_before[string] = string in memo
            earlier = np.array([read_before[string] for string in strings])
            ones = np.asarray(memo.membership_queries(strings), dtype=bool)
            counts[side] += (len(strings), earlier.sum(), (ones & ~earlier).sum())
        lo, hi = advantage_bounds(counts, level)
        read_low = sum(
            _clopper_pearson(read, drawn, level / 4)[0] for drawn, read, _ in counts
        )
        print(
            f"  certificate look {look}: {size} strings a side, advantage in "
            f"[{lo:.4f}, {hi:.4f}] against {target:.4f}"
        )
        if lo >= target:
            return True
        if read_low > slack:
            raise TooMuchRead(
                f"at least {read_low:.3f} of the strings on the two sides of the "
                f"DFA were read before, over the certificate's slack of {slack:.3f}"
            )
        if hi < target or hi - lo <= slack:
            return False
        size *= 2
        look += 1


def _given_label(pst, dfa, accepting):
    """(draw, mass): a draw from the sampler's strings given that dfa labels them
    accepting, and the sampler's weight on those strings, up to the scale of
    its symbol weights."""
    weights = pst.sampler.symbol_weights(pst.alphabet_size)
    length = pst.sampler.length
    states = sorted(q for q in dfa.states if (q in dfa.final_states) == accepting)
    paths = [count_paths_to_state(dfa, q, length, weights) for q in states]
    mass = np.array([float(p[length][dfa.initial_state]) for p in paths])
    total = float(mass.sum())
    if not total:
        return None, 0.0
    mass /= total

    def draw():
        state = pst.rng.choice(len(states), p=mass)
        return sample_string_reaching_state(dfa, paths[state], pst.rng, weights)

    return draw, total
