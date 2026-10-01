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

import numpy as np


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
