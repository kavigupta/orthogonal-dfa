"""Whether a hypothesis's error against the noiseless target is certified.

For a DFA h and the oracle's read O of a string x drawn by the sampler,

    A(h) = P(O = 1 | h(x) = 1) - P(O = 1 | h(x) = 0).

Noise that depends only on the label makes A(h) = (p_1 - p_0) A_f(h), with A_f
the same difference against the noiseless labels f, and

    P(h(x) != f(x)) <= max(q, 1 - q) (1 - A_f(h)),   q = P(h(x) = 1),

so where p_1 - p_0 = 2 s, A(h) > 2 s (1 - e / max(q, 1 - q)) holds h's error
within e.

The strings of each side are drawn as the state-reaching walks draw them, each
position independently by the sampler's symbol weights.
"""

import math

import numpy as np
import scipy.stats

from .dfa_utils import count_paths_to_state, sample_string_reaching_state


def clears(ones, drawn, gap, level) -> bool:
    """True only if, with d = ones[0] - ones[1],

        P(K_1 - K_0 >= d) <= level

    for every independent K_1 ~ Bin(drawn, p_1) and K_0 ~ Bin(drawn, p_0) with
    p_1 - p_0 <= gap.

    K_1 + (drawn - K_0) is a sum of 2 drawn Bernoullis whose rates sum to at most
    drawn (1 + gap).  Above its mean its tail is largest when the rates are
    equal (Hoeffding 1956, Theorem 4), so for d >= drawn gap + 1 the left side
    is at most P(Bin(2 drawn, (1 + gap) / 2) >= d + drawn).  Below that nothing
    is claimed, and the test fails."""
    difference = int(ones[0]) - int(ones[1])
    if difference < drawn * gap + 1:
        return False
    tail = scipy.stats.binom.sf(difference + drawn - 1, 2 * drawn, (1 + gap) / 2)
    return bool(tail <= level)


def look_level(alpha, look) -> float:
    """alpha 6 / (pi (look + 1))^2, which sums to alpha over every look."""
    return alpha * 6 / (math.pi * (look + 1)) ** 2


def first_look(gap, level) -> int:
    """The least n at which ones = (n, 0) clears gap at level."""
    return max(
        math.ceil(1 / (1 - gap)),
        math.ceil(math.log(level) / (2 * math.log((1 + gap) / 2))),
    )


def certifies(pst, dfa, *, alpha) -> bool:
    """True only with

        P(certifies and A(dfa) <= 2 s - d) <= alpha,   d = 2 s e / max(q, 1 - q),

    s = min_signal_strength, e = certified_error, q = P(dfa(x) = 1): where s is
    the oracle's signal exactly, a certified dfa's error is within e.

    Look k draws strings on each side of dfa's label up to n_k, doubling from
    first_look, and passes if their ones clear the gap 2 s - d at level
    look_level(alpha, k); those levels sum to alpha.  It refuses once clears,
    on the two sides' zeros, puts A(dfa) below 2 s."""
    (draw_1, mass_1), (draw_0, mass_0) = (
        _given_label(pst, dfa, True),
        _given_label(pst, dfa, False),
    )
    if not mass_1 or not mass_0:
        return False
    signal = pst.config.min_signal_strength
    accepted = mass_1 / (mass_1 + mass_0)
    gap = 2 * signal * (1 - pst.config.certified_error / max(accepted, 1 - accepted))
    ones = np.zeros(2, dtype=int)
    drawn = 0
    size = first_look(gap, look_level(alpha, 0))
    look = 0
    while True:
        level = look_level(alpha, look)
        for side, draw in enumerate((draw_1, draw_0)):
            strings = [draw() for _ in range(size - drawn)]
            ones[side] += sum(pst.oracle.membership_queries(strings))
        drawn = size
        print(
            f"  certificate look {look}: {drawn} strings a side, advantage "
            f"{(ones[0] - ones[1]) / drawn:.4f} against {gap:.4f}"
        )
        if clears(ones, drawn, gap, level):
            return True
        # The mirror of clears, which carries no guarantee: refusing only costs a
        # round.
        if clears(drawn - ones, drawn, -2 * signal, level):
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
