from typing import List, Tuple

import numpy as np

from .statistics import evidence_margin_for_population_size, give_up_check


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
    cluster = [seed_local]
    loss = float("inf")
    while True:
        cluster_center = masks[cluster].mean(0) > decision_boundary
        losses = (masks != cluster_center).sum(1)
        cluster = losses.argsort()[:count]
        if seed_local not in cluster:
            cluster = np.concatenate([[seed_local], cluster[: count - 1]])
        new_loss = losses[cluster].sum()
        if new_loss >= loss:
            break
        loss = new_loss

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
    min_signal_strength, suffix_family_size, decision_boundary, decision_rule_fpr
):
    result = evidence_margin_for_population_size(
        min_signal_strength,
        decision_rule_fpr,
        0.01,
        suffix_family_size,
        center=decision_boundary,
    )
    if result is None:
        return min_signal_strength * 0.5
    _, eps = result
    return eps


class GaveUpOnSuffixSearch(Exception):
    """Raised when the suffix search exceeds the give-up threshold."""


def sample_suffix_family(pst, v: int, first_round: bool) -> Tuple[List[int], float]:
    prev_fnr = 1.0
    strategy = "suffix"
    decision_boundary = pst.decision_boundary

    config = pst.config

    while True:
        seed_mask = pst.table.column(v)
        empirical_pos = float(seed_mask.mean())
        if first_round:
            _give_up_check(pst, config, seed_mask, empirical_pos)
        vs, decision_boundary = identify_cluster_around(
            pst, v, pst.config.suffix_family_size, decision_boundary
        )
        pst.decision_boundary = decision_boundary
        pst.evidence_margin = recompute_evidence_margin(
            pst.config.min_signal_strength,
            pst.config.suffix_family_size,
            decision_boundary,
            pst.config.decision_rule_fpr,
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
            pst.sample_more_suffixes(amount=pst.config.suffix_family_size)
        else:
            pst.sample_more_prefixes()


def _give_up_check(pst, config, seed_mask, empirical_pos):
    # Judge whether the signal is too weak over the representative prefixes only:
    # the short prefix-closed core explores a skewed region of the state space and
    # is classified more confidently than the probe prefixes, so folding it in
    # would distort the agreement estimate and could trigger a spurious give-up.
    rep = pst.table.representative
    seed_mask = seed_mask[rep]
    empirical_pos = float(seed_mask.mean())
    # The give-up statistic reasons about the sampled acceptance-family suffixes
    # (their count bounds how many are idempotent), so count the fully-observed
    # family suffixes -- not every interned row, which would also include
    # transition distinguishers.
    candidate = pst.table.fully_observed()
    result = give_up_check(
        config.min_signal_strength,
        int(rep.sum()),
        len(candidate),
        config.min_suffix_frequency,
        config.min_acc_rej,
        empirical_pos,
    )
    if result is not None:
        k, threshold = result
        masks = pst.table.observed_masks(candidate, rep)
        agreements = (masks == seed_mask).mean(axis=1)
        top_k_mean = float(np.sort(agreements)[-k:].mean())
        if top_k_mean <= threshold:
            raise GaveUpOnSuffixSearch(
                f"Sampled {len(candidate)} suffixes. "
                f"Top-{k} mean agreement {top_k_mean:.3f} <= {threshold:.3f}"
            )
