import numpy as np

STOPS = [(3, 0, 2), (3, 0, 0), (3, 2, 0)]


def closed_frames(arr):
    """``closed[i, f]`` iff a stop codon starts in row ``i`` at a position = f mod 3."""
    is_stop = np.zeros((arr.shape[0], arr.shape[1] - 2), bool)
    for a, b, c in STOPS:
        is_stop |= (arr[:, :-2] == a) & (arr[:, 1:-1] == b) & (arr[:, 2:] == c)
    return np.stack([is_stop[:, f::3].any(1) for f in range(3)], 1)
