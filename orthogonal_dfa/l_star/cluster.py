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


#: Rejections in a row after which no accept-preserving family is believed to
#: exist.  More suffixes is the only remedy, and none help against a target where
#: no suffix preserves the accept/reject classes.
ACCEPT_PRESERVING_GIVE_UP = 20

#: Significance at which a family is held not to be accept-preserving.  A
#: rejection costs a resample rather than a failure, so this is a resample budget
#: and not an error rate.
ACCEPT_PRESERVING_ALPHA = 1e-3


class NoAcceptPreservingFamily(Exception):
    """No accept-preserving suffix family could be sampled for this target."""


def accept_preserving_pvalue(pst, decision, decision_boundary, seed_row) -> float:
    """Exact two-sided binomial test that the family's cut is accept-preserving.

    Membership of ``p + v`` is membership of ``p`` for the empty suffix, so its
    column is the accept-preserving split itself: under the null it reads 1 at the
    accept rate on the prefixes the family accepts and at the reject rate on the
    rest.  The rates are unknown, and ``min_signal_strength`` bounds both away
    from the boundary -- a bound only makes the test reject less.
    """
    eps = pst.table.column(seed_row)[pst.table.representative]
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
    rejections = 0

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

        decision = pst.compute_decision(vs, pst.table.representative)
        # An undersized family is rejected either way, and counting its rejection
        # would spend the give-up budget, which means "no accept-preserving family
        # exists" and not "this one was too small".
        undersized = len(vs) < pst.config.suffix_family_size
        fnr = 1 if undersized else pst.fnr_from_decision(decision)
        preserving_p = (
            accept_preserving_pvalue(pst, decision, decision_boundary, v)
            if pst.config.require_accept_preserving and not undersized
            else 1.0
        )
        if preserving_p < ACCEPT_PRESERVING_ALPHA:
            rejections += 1
            if rejections >= ACCEPT_PRESERVING_GIVE_UP:
                raise NoAcceptPreservingFamily(
                    f"{rejections} families running were not accept-preserving "
                    f"(last p={preserving_p:.2e}); no family of "
                    f"{pst.config.suffix_family_size} suffixes realises the "
                    f"accept-preserving split on this target"
                )
            print(f"family is not accept-preserving (p={preserving_p:.2e}), resampling")
        else:
            rejections = 0
            if fnr <= pst.config.fnr_limit:
                print(
                    f"FNR limit reached, decision boundary: {decision_boundary:.4f}, "
                    f"margin: {pst.evidence_margin:.4f}"
                )
                return vs, decision_boundary

        if fnr >= prev_fnr or strategy == "prefix":
            strategy = "prefix" if strategy == "suffix" else "suffix"

        prev_fnr = fnr

        if preserving_p >= ACCEPT_PRESERVING_ALPHA:
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
