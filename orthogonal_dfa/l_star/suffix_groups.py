"""Groups of suffixes' rows of reads that differ by no more than read noise."""

from typing import List

import numpy as np
import scipy.stats
from sklearn.cluster import AgglomerativeClustering, KMeans


def coherent_groups(rows, k, alpha, rng) -> List[np.ndarray]:
    """The rows' k-means clusters over the even columns, unioned by complete
    linkage while every pair (a, b) across two unions has

        T = sum_p (mean_a,p - mean_b,p)^2 / (v_p (1 / n_a + 1 / n_b))

    over the odd columns p within the chi-squared quantile at 1 - alpha / pairs,
    v_p the within-cluster variance; then each row moved to its nearest union
    over the even columns.  The clusters never see the odd columns, so under one
    profile T is chi-squared, the means taken as normal."""
    fit, held = rows[:, ::2], rows[:, 1::2]
    k = min(k, len(np.unique(fit, axis=0)))
    seed = int(rng.integers(2**31))
    _, label = np.unique(
        KMeans(k, n_init=1, random_state=seed).fit_predict(fit), return_inverse=True
    )
    k = label.max() + 1
    means = np.array([held[label == g].mean(0) for g in range(k)])
    variance = ((held - means[label]) ** 2).sum(0) / max(1, len(rows) - k)
    readable = variance > 0
    if k > 1 and readable.any():
        gaps = ((means[:, None] - means[None]) ** 2)[..., readable] / variance[readable]
        sizes = np.bincount(label)
        statistic = gaps.sum(2) / (1 / sizes[:, None] + 1 / sizes[None])
        label = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="complete",
            distance_threshold=scipy.stats.chi2.isf(
                alpha / (k * (k - 1) / 2), readable.sum()
            ),
        ).fit_predict(statistic)[label]
    centers = np.array([fit[label == u].mean(0) for u in np.unique(label)])
    label = KMeans(len(centers), init=centers, n_init=1).fit_predict(fit)
    return [np.flatnonzero(label == g) for g in np.unique(label)]


def nearest_to_anchor_group(rows, anchor, k, alpha, rng) -> np.ndarray:
    """Every row, nearest first to the mean of the coherent_group whose rows
    covary most with anchor on average.  The groups are found without anchor and
    the mean is that group's alone, so rows that share a misreading cannot pull
    the order toward themselves."""
    groups = coherent_groups(rows, k, alpha, rng)
    centred = anchor - anchor.mean()
    best = max(groups, key=lambda g: (rows[g] @ centred).mean())
    return np.argsort(((rows - rows[best].mean(0)) ** 2).sum(1), kind="stable")
