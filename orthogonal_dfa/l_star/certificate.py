"""Whether a hypothesis carries the share of the promised signal it is held to.

For a DFA h and the oracle's read O of a string x drawn by the sampler,

    A(h) = P(O = 1 | h(x) = 1) - P(O = 1 | h(x) = 0).

Noise that depends only on the label makes A(h) = (p_1 - p_0) A_f(h), with A_f
the same difference against the noiseless labels, so a promised signal s puts
A(target) >= 2 s, and a state of mass m that h labels wrongly lowers A_f by m / q
or m / (1 - q), q = P(h(x) = 1).

The strings of each side are drawn as the state-reaching walks draw them, each
position independently by the sampler's symbol weights.
"""

import math
from typing import Tuple

import numpy as np
import scipy.stats

from .dfa_utils import count_paths_to_state, sample_string_reaching_state


def advantage_bounds(sides, level) -> Tuple[float, float]:
    """(lo, hi) with P(A < lo) <= level and P(A > hi) <= level, for sides[s] =
    (n, k, o): of n strings drawn given h = s, k were read before the draw, and o
    of the other n - k read 1.  With m_s = P(read before | h = s), u_s the rate
    of O = 1 on the rest, and O unconstrained on what was read before,

        (1 - m_1) u_1 - u_0 - m_0 (1 - u_0) <= A <= u_1 + m_1 (1 - u_1) - (1 - m_0) u_0,

    each side of which is bounded by Clopper-Pearson bounds at level / 4 on the
    four rates it holds, joined by a union bound."""
    (drawn_1, read_1, ones_1), (drawn_0, read_0, ones_0) = sides
    quarter = level / 4
    _, read_high_1 = _clopper_pearson(read_1, drawn_1, quarter)
    _, read_high_0 = _clopper_pearson(read_0, drawn_0, quarter)
    low_1, high_1 = _clopper_pearson(ones_1, drawn_1 - read_1, quarter)
    low_0, high_0 = _clopper_pearson(ones_0, drawn_0 - read_0, quarter)
    lo = (1 - read_high_1) * low_1 - high_0 - read_high_0 * (1 - high_0)
    hi = high_1 + read_high_1 * (1 - high_1) - (1 - read_high_0) * low_0
    return lo, hi


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

        A(dfa) >= 2 s c,   s = min_signal_strength, c = certified_signal_share,

    before they put it below, narrow to 2 s (1 - c) apart, or show more of the
    two sides read before than that slack; m_k doubles from first_look at look
    0's level.  P(certifies and A(dfa) < 2 s c) <= alpha."""
    sides = [_given_label(pst, dfa, True), _given_label(pst, dfa, False)]
    if None in sides:
        return False
    signal = pst.config.min_signal_strength
    share = pst.config.certified_signal_share
    target = 2 * signal * share
    slack = 2 * signal * (1 - share)
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
        if hi < target or hi - lo <= slack or read_low > slack:
            return False
        size *= 2
        look += 1


def _given_label(pst, dfa, accepting):
    """A draw from the sampler's strings given that dfa labels them accepting,
    or None if the sampler puts no mass there."""
    weights = pst.sampler.symbol_weights(pst.alphabet_size)
    length = pst.sampler.length
    states = sorted(q for q in dfa.states if (q in dfa.final_states) == accepting)
    paths = [count_paths_to_state(dfa, q, length, weights) for q in states]
    mass = np.array([float(p[length][dfa.initial_state]) for p in paths])
    if not mass.sum():
        return None
    mass /= mass.sum()

    def draw():
        state = pst.rng.choice(len(states), p=mass)
        return sample_string_reaching_state(dfa, paths[state], pst.rng, weights)

    return draw
