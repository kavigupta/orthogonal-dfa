from typing import List, Tuple

import numpy as np

from .statistics import (
    evidence_margin_for_population_size,
    max_suffixes_before_giving_up,
)


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


def sample_suffix_family(pst, v: int) -> Tuple[List[int], float]:
    prev_fnr = 1.0
    strategy = "suffix"
    decision_boundary = pst.decision_boundary

    config = pst.config
    max_suffixes = None
    agreement_threshold = None
    exceedance_count_threshold = None

    while True:
        if config.min_suffix_frequency is not None:
            max_suffixes, agreement_threshold, exceedance_count_threshold = (
                max_suffixes_before_giving_up(
                    config.min_signal_strength,
                    len(pst.prefixes),
                    config.min_suffix_frequency,
                )
            )
            print(
                f"Max suffixes before giving up: {max_suffixes}, "
                f"agreement threshold: {agreement_threshold}/{len(pst.prefixes)}, "
            )
        if max_suffixes is not None and len(pst.suffix_bank) >= max_suffixes:
            seed_mask = np.array(pst.corresponding_masks[v])
            masks = np.array(pst.corresponding_masks)
            agreements = (masks == seed_mask).sum(axis=1)
            exceeding = int((agreements > agreement_threshold).sum())
            if exceeding <= exceedance_count_threshold:
                raise GaveUpOnSuffixSearch(
                    f"Sampled {len(pst.suffix_bank)} suffixes (limit {max_suffixes}). "
                    f"Exceedances: {exceeding} <= {exceedance_count_threshold} "
                    f"(agreement threshold: {agreement_threshold})"
                )
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
