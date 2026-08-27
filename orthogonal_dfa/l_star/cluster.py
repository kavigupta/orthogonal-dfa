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

    # A cluster all on one side estimates a boundary whose implied rates,
    # boundary +/- the signal, are no longer probabilities.
    signal = pst.config.min_signal_strength
    decision_boundary = min(max(decision_boundary, signal), 1 - signal)

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

#: Confidence the drift bounds below are read at, the same rate the population
#: sizing is designed to.
ACCEPT_PRESERVING_CONFIDENCE = 0.01

#: What the gate was able to conclude about a family.
ADMITTED, DRIFTED, UNCERTIFIED = "admitted", "drifted", "uncertified"


class NoAcceptPreservingFamily(Exception):
    """No accept-preserving suffix family could be sampled for this target."""


def tolerated_drift(min_signal_strength: float, evidence_margin: float) -> float:
    """Share of a side that may be the other class before the family stops reading
    its own classes correctly.

    A side carrying a fraction ``f`` of the other class reads, on the empty suffix,

        boundary + signal * (1 - 2 * f)

    and the decision only calls that side accepting while it stays at or past
    ``boundary + evidence_margin``, which bounds f.
    """
    return (min_signal_strength - evidence_margin) / (2 * min_signal_strength)


def _rate_interval(hits: int, n: int) -> Tuple[float, float]:
    """Clopper-Pearson interval for ``hits``/``n``."""
    conf = ACCEPT_PRESERVING_CONFIDENCE
    low = scipy.stats.beta.ppf(conf, hits, n - hits + 1) if hits else 0.0
    high = scipy.stats.beta.ppf(1 - conf, hits + 1, n - hits) if hits < n else 1.0
    return float(low), float(high)


def accept_preserving_drift(
    pst, decision, decision_boundary, seed_row
) -> Tuple[float, float]:
    """Interval on the share of a side that is the other class, worst side first.

    Membership of ``p + v`` is membership of ``p`` for the empty suffix, so its
    column is the accept-preserving split itself: a family realising the split
    reads 1 there at the accept rate on the prefixes it accepts and at the reject
    rate on the rest.  Mixing the classes moves that rate toward the boundary by
    an amount linear in the share mixed in, so an interval on the rate inverts to
    one on the share.

    An interval is a statement about the family.  Failing to reject a null is only
    a statement about the evidence, which is how a family too small to measure
    used to pass for one that had been checked.
    """
    column = pst.table.column(seed_row)[pst.table.representative]
    signal = pst.config.min_signal_strength
    low = high = 0.0
    read = False
    for side, rate, mixes_down in (
        (decision >= pst.accept_thresh, decision_boundary + signal, True),
        (decision < pst.reject_thresh, decision_boundary - signal, False),
    ):
        n = int(side.sum())
        # A rate at 0 or 1 is the boundary sitting on its clamp, where the side
        # carries no evidence either way. Read the other one rather than hold the
        # family against a side nothing could ever certify.
        if n == 0 or not 0 < rate < 1:
            continue
        read = True
        rate_low, rate_high = _rate_interval(int(column[side].sum()), n)
        if mixes_down:
            side_low, side_high = rate - rate_high, rate - rate_low
        else:
            side_low, side_high = rate_low - rate, rate_high - rate
        low = max(low, side_low / (2 * signal))
        high = max(high, side_high / (2 * signal))
    # Neither side readable: uncertified, rather than certified clean.
    if not read:
        return 0.0, 1.0
    return max(0.0, low), min(1.0, high)


class AcceptPreservingGate:
    """Holds each suffix family to the accept-preserving split, across the loop
    that resamples until one passes.  Spends the give-up budget only on families
    shown to have drifted, never on ones there was not the evidence to judge."""

    def __init__(self, config):
        self.enabled = config.require_accept_preserving
        self.rejections = 0

    def verdict(self, pst, decision, decision_boundary, seed_row) -> str:
        if not self.enabled:
            return ADMITTED
        low, high = accept_preserving_drift(pst, decision, decision_boundary, seed_row)
        tolerated = tolerated_drift(pst.config.min_signal_strength, pst.evidence_margin)
        if high <= tolerated:
            self.rejections = 0
            return ADMITTED
        # Drift this far is consistent with the evidence but so is none of it.
        if low <= tolerated:
            return UNCERTIFIED
        self.rejections += 1
        if self.rejections >= ACCEPT_PRESERVING_GIVE_UP:
            raise NoAcceptPreservingFamily(
                f"{self.rejections} families running carried at least "
                f"{low:.0%} of each class on the other's side, past the "
                f"{tolerated:.0%} the decision can absorb; no suffix family "
                f"realises the accept-preserving split on this target"
            )
        return DRIFTED


def sample_suffix_family(pst, v: int) -> Tuple[List[int], float]:
    prev_effective_fnr = 1.0
    strategy = "suffix"
    decision_boundary = pst.decision_boundary
    gate = AcceptPreservingGate(pst.config)

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
        fnr = pst.fnr_from_decision(decision)
        effective_fnr, reason = fnr, f"FNR {fnr:.4f} too high"
        # An undersized family is unusable whatever its FNR measures, and testing
        # it would spend a budget that means no accept-preserving family exists.
        verdict = ADMITTED
        if len(vs) < pst.config.suffix_family_size:
            effective_fnr, reason = 1.0, "undersized"
        else:
            verdict = gate.verdict(pst, decision, decision_boundary, v)
            if verdict is DRIFTED:
                effective_fnr, reason = 1.0, "not accept-preserving"
            elif verdict is UNCERTIFIED:
                effective_fnr, reason = 1.0, "accept-preserving not established"

        if effective_fnr <= pst.config.fnr_limit:
            print(
                f"FNR limit reached, decision boundary: {decision_boundary:.4f}, "
                f"margin: {pst.evidence_margin:.4f}"
            )
            return vs, decision_boundary

        if verdict is UNCERTIFIED:
            # The bound is wide because the prefixes are few, and prefixes are
            # what the split is read against.
            strategy = "prefix"
        elif effective_fnr >= prev_effective_fnr or strategy == "prefix":
            strategy = "prefix" if strategy == "suffix" else "suffix"

        prev_effective_fnr = effective_fnr

        print(
            f"{reason}, sampling more {strategy}es; "
            f"decision_boundary: {decision_boundary:.4f}"
        )

        if strategy == "suffix":
            kept = pst.sample_more_suffixes(
                amount=pst.config.suffix_family_size, reference=v
            )
            print(f"  kept {kept}/{pst.config.suffix_family_size} after screening")
        else:
            pst.sample_more_prefixes()
