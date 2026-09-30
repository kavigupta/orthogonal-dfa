"""Whether a hypothesis carries the share of the promised signal it is held to.

For a DFA h and the oracle's read O of a string x drawn by the sampler,

    A(h) = P(O = 1 | h(x) = 1) - P(O = 1 | h(x) = 0).

Noise that depends only on the label makes A(h) = (p_1 - p_0) A_f(h), with A_f
the same difference against the noiseless labels, so a promised signal s puts
A(target) >= 2 s, and every state h merges into the wrong label costs A(h) its
share of that.
"""

import math
from typing import Tuple

import numpy as np
import scipy.stats

from .dfa_utils import count_paths_to_state, sample_string_reaching_state


def advantage_bounds(counts, level) -> Tuple[float, float]:
    """(lo, hi) with P(A < lo) <= level and P(A > hi) <= level, for counts =
    ((k_1, n_1), (k_0, n_0)), k_s of the n_s strings with h = s reading 1:
    Clopper-Pearson bounds at level / 2 on each side's rate, joined by a union
    bound."""
    (ones, drawn), (zeros_ones, zeros_drawn) = counts
    low_one, high_one = _clopper_pearson(ones, drawn, level / 2)
    low_zero, high_zero = _clopper_pearson(zeros_ones, zeros_drawn, level / 2)
    return low_one - high_zero, high_one - low_zero


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
    """The least m with 2 (level / 2)^(1 / m) - 1 >= target: below it not even m
    strings a side, every one read as h reads it, clear the target at level."""
    return math.ceil(math.log(level / 2) / math.log((1 + target) / 2))


def certifies(pst, dfa, *, alpha) -> bool:
    """Whether advantage_bounds, on m_k unread strings a side at look k drawn by
    the sampler given dfa's label, at level look_level(alpha, k), put

        A(dfa) >= 2 s c,   s = min_signal_strength, c = certified_signal_share,

    before they put it below, or narrow to 2 s (1 - c) apart; m_k doubles from
    first_look at look 0's level."""
    sides = [_given_label(pst, dfa, True), _given_label(pst, dfa, False)]
    if None in sides:
        return False
    signal = pst.config.min_signal_strength
    share = pst.config.certified_signal_share
    target = 2 * signal * share
    counts = np.zeros((2, 2), dtype=int)
    size = first_look(target, look_level(alpha, 0))
    look = 0
    while True:
        for side, draw in enumerate(sides):
            strings = _unread(draw, pst.table.memo, size - counts[side, 1])
            reads = pst.table.memo.membership_queries(strings)
            counts[side] += (sum(reads), len(strings))
        lo, hi = advantage_bounds(counts, look_level(alpha, look))
        print(
            f"  certificate look {look}: {size} strings a side, advantage in "
            f"[{lo:.4f}, {hi:.4f}] against {target:.4f}"
        )
        if lo >= target:
            return True
        if hi < target or hi - lo <= 2 * signal * (1 - share):
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


def _unread(draw, memo, count):
    """count distinct strings from draw that memo has not read."""
    drawn = set()
    while len(drawn) < count:
        string = draw()
        if string not in memo:
            drawn.add(string)
    return sorted(drawn)
