from typing import List, Tuple

import numpy as np

from .statistics import evidence_margin_for_population_size, give_up_check


def identify_cluster_around(
    pst, seed: int, count: int, decision_boundary: float
) -> Tuple[List[int], float]:
    masks = np.array(pst.corresponding_masks)
    cluster = [seed]
    loss = float("inf")
    while True:
        cluster_center = masks[cluster].mean(0) > decision_boundary
        losses = (masks != cluster_center).sum(1)
        cluster = losses.argsort()[:count]
        if seed not in cluster:
            cluster = np.concatenate([[seed], cluster[: count - 1]])
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

    return cluster.tolist(), decision_boundary


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


class GaveUpOnSuffixSearch(Exception):
    """Raised when the suffix search exceeds the give-up threshold."""


def sample_suffix_family(pst, v: int, first_round: bool) -> Tuple[List[int], float]:
    prev_fnr = 1.0
    strategy = "suffix"
    decision_boundary = pst.decision_boundary

    config = pst.config

    while True:
        seed_mask = np.array(pst.corresponding_masks[v])
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
    result = give_up_check(
        config.min_signal_strength,
        len(pst.prefixes),
        len(pst.suffix_bank),
        config.min_suffix_frequency,
        config.min_acc_rej,
        empirical_pos,
    )
    if result is not None:
        k, threshold = result
        masks = np.array(pst.corresponding_masks)
        agreements = (masks == seed_mask).mean(axis=1)
        top_k_mean = float(np.sort(agreements)[-k:].mean())
        if top_k_mean <= threshold:
            raise GaveUpOnSuffixSearch(
                f"Sampled {len(pst.suffix_bank)} suffixes. "
                f"Top-{k} mean agreement {top_k_mean:.3f} <= {threshold:.3f}"
            )
