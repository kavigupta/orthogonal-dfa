"""Groups of rows, suffixes' reads over the prefixes, that read alike up to
read noise."""

from typing import List

import numpy as np
import scipy.stats
from sklearn.cluster import AgglomerativeClustering, KMeans


def _merged(held, groups, alpha) -> np.ndarray:
    """Each group's union under complete linkage, joining two unions while every
    pair (a, b) across them has

        T = sum_p (mean_a,p - mean_b,p)^2 / (v_p (1 / n_a + 1 / n_b))

    below the chi-squared quantile at 1 - alpha / pairs, over the columns p of
    held, one degree of freedom each, v_p the within-group variance of column p
    and pairs every pair of groups: under one profile T is chi-squared, the
    means taken as normal, so long as the groups were formed without held.
    Each T is taken once, on the groups as given; a union's own mean is never
    tested, since which groups it holds was chosen on held."""
    if len(groups) == 1:
        return np.zeros(1, dtype=int)
    within = sum(((held[g] - held[g].mean(0)) ** 2).sum(0) for g in groups)
    variance = within / max(1, len(held) - len(groups))
    readable = variance > 0
    means = np.array([held[g].mean(0) for g in groups])[:, readable]
    sizes = np.array([len(g) for g in groups])
    gaps = ((means[:, None] - means[None]) ** 2 / variance[readable]).sum(2)
    statistic = gaps / (1 / sizes[:, None] + 1 / sizes[None])
    pairs = len(groups) * (len(groups) - 1) // 2
    return AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="complete",
        distance_threshold=scipy.stats.chi2.isf(alpha / pairs, readable.sum()),
    ).fit_predict(statistic)


def coherent_groups(rows, k, alpha, rng) -> List[np.ndarray]:
    """The rows' k-means clusters over the even columns, _merged over the odd
    ones, each row then moved to its nearest union over the even columns: a
    cluster that straddles two profiles joins neither, and its rows rejoin the
    one they are nearest."""
    fit, held = rows[:, ::2], rows[:, 1::2]
    seed = int(rng.integers(2**31))
    distinct = len(np.unique(fit, axis=0))
    label = KMeans(min(k, distinct), n_init=1, random_state=seed).fit_predict(fit)
    _, label = np.unique(label, return_inverse=True)
    groups = [np.flatnonzero(label == g) for g in range(label.max() + 1)]
    union = _merged(held, groups, alpha)[label]
    centers = np.array([fit[union == u].mean(0) for u in np.unique(union)])
    label = KMeans(len(centers), init=centers, n_init=1).fit_predict(fit)
    return [np.flatnonzero(label == g) for g in np.unique(label)]
