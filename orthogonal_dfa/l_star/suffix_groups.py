"""Groups of suffixes' rows of reads that differ by no more than read noise, and
the rows ranked from the group that reads like an anchor.

Columns are prefixes and populations are masks over them.  Deciding which rows
share a profile uses every column once, since how much a test can see is a
matter of counts; the ranking weighs each population the same, since it
estimates a property of the populations' distributions.
"""

from typing import List

import numpy as np
import scipy.stats
from sklearn.cluster import AgglomerativeClustering, KMeans


def coherent_groups(rows, populations, *, k, alpha, rng) -> List[np.ndarray]:
    """The rows' k-means clusters over the even columns, unioned by complete
    linkage while no pair (a, b) across two unions has, for j every column or
    one population's,

        T_j = sum_p (mean_a,p - mean_b,p)^2 / (v_p (1 / n_a + 1 / n_b))

    over j's odd columns p beyond the chi-squared quantile at
    1 - alpha / (pairs (|populations| + 1)), v_p the within-cluster variance;
    then each row moved to its nearest union over the even columns.  The
    clusters never see the odd columns, so under one profile T_j is
    chi-squared, the means taken as normal.  Every column at once catches a
    difference spread thinly across populations; one population's columns
    catch a difference within it however many other populations hold."""
    fit, held = rows[:, ::2], rows[:, 1::2]
    k = min(k, len(np.unique(fit, axis=0)))
    seed = int(rng.integers(2**31))
    _, label = np.unique(
        KMeans(k, n_init=1, random_state=seed).fit_predict(fit), return_inverse=True
    )
    k = label.max() + 1
    if k > 1:
        means = np.array([held[label == g].mean(0) for g in range(k)])
        variance = ((held - means[label]) ** 2).sum(0) / max(1, len(rows) - k)
        sizes = np.bincount(label)
        # Every column at once too, for a difference spread thinly across them.
        tested = [np.ones(rows.shape[1], dtype=bool), *populations]
        level = alpha / (len(tested) * k * (k - 1) / 2)
        rejected = np.zeros((k, k), dtype=bool)
        for population in tested:
            columns = population[1::2] & (variance > 0)
            if columns.any():
                gaps = (means[:, None, columns] - means[None, :, columns]) ** 2
                statistic = (gaps / variance[columns]).sum(2) / (
                    1 / sizes[:, None] + 1 / sizes[None]
                )
                rejected |= statistic > scipy.stats.chi2.isf(level, columns.sum())
        label = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="complete",
            distance_threshold=1 / 2,
        ).fit_predict(rejected.astype(float))[label]
    centers = np.array([fit[label == u].mean(0) for u in np.unique(label)])
    label = KMeans(len(centers), init=centers, n_init=1).fit_predict(fit)
    return [np.flatnonzero(label == g) for g in np.unique(label)]


def misread_statistic(rows, anchor, columns, rates) -> np.ndarray:
    """Per row v, with rates = (p_0, p_1) and y the anchor's reads,

        U(v) = sum_p [ (x_vp - y_p)^2 - 2 p_0 p_1 - (1 - p_0 - p_1)(x_vp + y_p) ]

    over the columns marked columns: expectation (p_1 - p_0)^2 times the
    columns where v and anchor read in different classes, since a column read
    at r by both adds -2 (r - p_0)(r - p_1) = 0."""
    p_0, p_1 = rates
    x, y = rows[:, columns], anchor[columns]
    ones = x.sum(1) + y.sum()
    return (p_0 + p_1) * ones - 2 * x @ y - 2 * columns.sum() * p_0 * p_1


def nearest_to_anchor_group(
    rows, anchor, populations, scoring, rates, *, k, alpha, rng
):
    """Every row, nearest first to the mean of the coherent_group whose rows
    have the least misread_statistic against anchor on average over the
    columns marked scoring, distance weighing each population's columns the
    same:

        d(v) = sum_p w_p (x_v,p - mean_p)^2,   w_p proportional to
               sum over populations j holding p of 1 / |j|.

    The groups are found without anchor and the mean is that group's alone, so
    rows that share a misreading cannot pull the order toward themselves."""
    groups = coherent_groups(rows, populations, k=k, alpha=alpha, rng=rng)
    misread = misread_statistic(rows, anchor, scoring, rates)
    best = min(groups, key=lambda g: misread[g].mean())
    weight = sum(population / population.sum() for population in populations)
    distance = (rows - rows[best].mean(0)) ** 2 @ weight
    return np.argsort(distance, kind="stable")


def aligned_suffixes(rows, anchor, populations, rates, *, epsilon, alpha):
    """The rows whose share of each population read in another class than
    anchor's is at most epsilon / 2 by the bound below, smallest largest bound
    first.  misread_statistic over population j's n_j columns has independent
    terms in an interval of width w, so with probability at least 1 - alpha over
    every row and population at once that share is at most

        (U_j + w sqrt(n_j log(rows populations / alpha) / 2)) / ((p_1 - p_0)^2 n_j).
    """
    if rows.shape[0] == 0 or not populations:
        return np.zeros(0, dtype=int)
    p_0, p_1 = rates
    # A column's term where both read 0, where one does, and where both read 1.
    terms = -2 * p_0 * p_1 + np.array([0, p_0 + p_1, 2 * (p_0 + p_1 - 1)])
    width = terms.max() - terms.min()
    log_tests = np.log(len(rows) * len(populations) / alpha)
    worst = np.zeros(len(rows))
    for population in populations:
        n = population.sum()
        u = misread_statistic(rows, anchor, population, rates)
        share = (u + width * np.sqrt(n * log_tests / 2)) / ((p_1 - p_0) ** 2 * n)
        worst = np.maximum(worst, share)
    kept = np.flatnonzero(worst <= epsilon / 2)
    return kept[np.argsort(worst[kept], kind="stable")]
