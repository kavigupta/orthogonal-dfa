from typing import List, Tuple

import numpy as np
import scipy.stats

from .statistics import evidence_margin_for_population_size


def identify_cluster_around(
    pst, seed: int, count: int, decision_boundary: float
) -> Tuple[List[int], float]:
    # Cluster only over fully-observed suffix columns -- the sampled acceptance-
    # family suffixes -- to avoid forcing a bunch of additional computation on the
    # partially-observed transition distinguishers.
    #
    # Restrict to representative (non-core) prefix columns: the suffix family and
    # the decision boundary are global calibration and must not be biased by the
    # statistically-unrepresentative short prefix-closed core.
    candidate = pst.table.fully_observed()
    masks = pst.table.observed_masks(candidate, pst.table.representative)
    seed_local = int(np.searchsorted(candidate, seed))
    assert candidate[seed_local] == seed, "cluster seed must be fully observed"
    # Only keep clustering while the seed belongs to the cluster.
    # We want to avoid drifting the cluster center away from the seed, which can
    # happen if the seed has a very small cluster relative to `count`.
    cluster = [seed_local]
    loss = float("inf")
    while True:
        cluster_center = masks[cluster].mean(0) > decision_boundary
        losses = (masks != cluster_center).sum(1)
        nearest = losses.argsort()[:count]
        if seed_local not in nearest:
            break
        new_loss = losses[nearest].sum()
        if new_loss >= loss:
            break
        cluster, loss = nearest, new_loss

    # Estimate decision boundary from the prefix separation
    prefix_means = masks[cluster].mean(0)
    accept_prefixes = prefix_means[cluster_center]
    reject_prefixes = prefix_means[~cluster_center]
    accept_mean = (
        accept_prefixes.mean() if len(accept_prefixes) > 0 else decision_boundary
    )
    reject_mean = (
        reject_prefixes.mean() if len(reject_prefixes) > 0 else decision_boundary
    )
    if len(accept_prefixes) > 0 and len(reject_prefixes) > 0:
        decision_boundary = (accept_mean + reject_mean) / 2
    elif len(accept_prefixes) > 0:
        # didn't find any rejects, so just put the boundary in the middle of the accepts
        decision_boundary = accept_mean
    elif len(reject_prefixes) > 0:
        # symmetric to above
        decision_boundary = reject_mean

    return candidate[cluster].tolist(), decision_boundary


def recompute_evidence_margin(
    min_signal_strength, suffix_family_size, decision_boundary
):
    result = evidence_margin_for_population_size(
        min_signal_strength, 0.01, 0.01, suffix_family_size, center=decision_boundary
    )
    if result is None:
        return min_signal_strength * 0.5
    _, eps = result
    return eps


#: Significance at which a family is held to disagree with epsilon.  A rejection
#: costs a resample rather than a failure, so this is set against the resample
#: budget -- the families a run evaluates -- not a family-wise error rate.
EPSILON_AUDIT_ALPHA = 1e-3


def epsilon_audit_pvalue(pst, vs, decision_boundary) -> float:
    """Exact two-sided binomial test of the family's cut against epsilon's column.

    Membership of ``p + eps`` is membership of ``p``, so epsilon's column *is* the
    accept-preserving split.  Under the null that the family realises that split,
    epsilon reads 1 at the accept rate on the prefixes the family accepts and at
    the reject rate on those it rejects.  A family that has settled into a
    neighbouring signature class -- still containing epsilon, but outvoting it --
    depletes the one and enriches the other, which the FNR gate cannot see because
    such a family is perfectly decisive.

    The true rates are unknown; ``min_signal_strength`` bounds them away from the
    boundary, and substituting a bound only makes the test conservative.
    """
    mask = pst.table.representative
    decision = pst.compute_decision(vs, mask)
    eps = pst.table.column(pst.table.intern_suffix([]))[mask]
    signal = pst.config.min_signal_strength
    worst = 1.0
    for side, rate in (
        (decision >= pst.accept_thresh, min(decision_boundary + signal, 1 - 1e-9)),
        (decision < pst.reject_thresh, max(decision_boundary - signal, 1e-9)),
    ):
        n = int(side.sum())
        if n < 2:
            continue
        hits = int(eps[side].sum())
        tail = (
            scipy.stats.binom.cdf(hits, n, rate)
            if rate > 0.5
            else scipy.stats.binom.sf(hits - 1, n, rate)
        )
        worst = min(worst, float(tail))
    return min(1.0, 2 * worst)


def sample_suffix_family(pst, v: int) -> Tuple[List[int], float]:
    prev_fnr = 1.0
    strategy = "suffix"
    decision_boundary = pst.decision_boundary

    while True:
        # Promotes the seed to fully observed, which identify_cluster_around
        # requires of it. Redone each round, since more prefixes may have
        # arrived since the last one.
        pst.table.column(v)
        vs, decision_boundary = identify_cluster_around(
            pst, v, pst.config.suffix_family_size, decision_boundary
        )
        pst.decision_boundary = decision_boundary
        pst.evidence_margin = recompute_evidence_margin(
            pst.config.min_signal_strength,
            pst.config.suffix_family_size,
            decision_boundary,
        )

        audit = epsilon_audit_pvalue(pst, vs, decision_boundary)
        fnr = 1 if len(vs) < pst.config.suffix_family_size else pst.compute_fnr(vs)
        if audit < EPSILON_AUDIT_ALPHA:
            print(f"family disagrees with epsilon (p={audit:.2e}), resampling")
        elif fnr <= pst.config.fnr_limit:
            print(
                f"FNR limit reached, decision boundary: {decision_boundary:.4f}, "
                f"margin: {pst.evidence_margin:.4f}"
            )
            return vs, decision_boundary

        if fnr >= prev_fnr or strategy == "prefix":
            strategy = "prefix" if strategy == "suffix" else "suffix"

        prev_fnr = fnr

        print(
            f"FNR {fnr:.4f} too high, sampling more suffixes; "
            f"decision_boundary: {decision_boundary:.4f}"
        )

        if strategy == "suffix":
            kept = pst.sample_more_suffixes(
                amount=pst.config.suffix_family_size, reference=v
            )
            print(f"  kept {kept}/{pst.config.suffix_family_size} after screening")
        else:
            pst.sample_more_prefixes()
