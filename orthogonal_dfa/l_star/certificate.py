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


def error_bound(masses, accepting, low, high, gap) -> float:
    """The largest sum_S m_S e_S, m_S = masses[S], over every r_S in
    [low_S, high_S] and every p_0 with each c_S = (r_S - p_0) / band in [0, 1],

        band = max(gap, max_S low_S - min_S high_S),

    the narrowest band at least gap that the rates fit.

    For a fixed p_0 each state's worst rate is independent of the others', so
    the error is piecewise linear in p_0, and the largest is at the ends of the
    p_0 that fit or where a state's worst rate meets a bound."""
    masses, low, high = (np.asarray(v, dtype=float) for v in (masses, low, high))
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
        return float(masses @ np.where(accepting, 1 - share, share))

    corners = np.concatenate([[least, most], high - band, low])
    return max(error(p) for p in corners if least <= p <= most)
