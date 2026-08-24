from typing import List, Tuple

import numpy as np

from .statistics import evidence_margin_for_population_size


def _converge(masks, available, seed_local, count, decision_boundary):
    """One clustering run from ``seed_local``, restricted to ``available`` rows."""
    cluster = np.array([seed_local])
    loss = float("inf")
    while True:
        cluster_center = masks[cluster].mean(0) > decision_boundary
        losses = (masks != cluster_center).sum(1).astype(float)
        losses[~available] = np.inf
        nearest = np.argsort(losses)[: min(count, int(available.sum()))]
        new_loss = losses[nearest].sum()
        if not np.isfinite(new_loss) or new_loss >= loss:
            return cluster
        cluster, loss = nearest, new_loss


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
    # Cluster as usual, but return the cluster epsilon actually belongs to.
    # Refinement can converge onto a group that agrees with itself and not with
    # epsilon; splicing epsilon back in gives a family whose cut disagrees with
    # it.  When that happens the converged group is simply a different cluster,
    # so peel it off and cluster again from epsilon over what is left.
    available = np.ones(len(candidate), dtype=bool)
    while True:
        cluster = _converge(masks, available, seed_local, count, decision_boundary)
        if seed_local in cluster or available.sum() <= count:
            break
        available[cluster] = False
    cluster_center = masks[cluster].mean(0) > decision_boundary

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

        fnr = 1 if len(vs) < pst.config.suffix_family_size else pst.compute_fnr(vs)
        if fnr <= pst.config.fnr_limit:
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
