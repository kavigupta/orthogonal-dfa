from typing import List, Tuple

import numpy as np
import scipy.stats

from .statistics import (
    evidence_margin_for_population_size,
    population_size_and_evidence_margin,
)


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


def smallest_readable_family(min_signal_strength, decision_boundary):
    """Fewest suffixes a decision at this boundary can be read over.

    How many it needs depends on where the boundary sits: the two classes draw
    from binomials whose variance differs once it leaves 0.5.
    """
    size, _ = population_size_and_evidence_margin(
        min_signal_strength, 0.01, 0.01, center=decision_boundary
    )
    return size


def readable_size_and_margin(min_signal_strength, decision_boundary, have, smallest):
    """The largest size at or below ``have`` whose band holds both error rates, and
    the margin that reads it.  ``have`` must be at least ``smallest``.

    Sizes just above the minimum can admit no band at all -- one more suffix shifts
    every operating point off the integer lattice -- so step down rather than call
    a family that is large enough undersized. ``smallest`` always admits one, so
    the walk cannot run off the end.
    """
    for size in range(have, smallest - 1, -1):
        found = evidence_margin_for_population_size(
            min_signal_strength, 0.01, 0.01, size, center=decision_boundary
        )
        if found is not None:
            return size, found[1]
    raise AssertionError(f"{smallest} suffixes was supposed to be readable")


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
    """Bonferroni over one one-sided binomial test per side of the family's cut.

    Membership of ``p + v`` is membership of ``p`` for the empty suffix, so its
    column is the accept-preserving split itself: under the null it reads 1 at the
    accept rate on the prefixes the family accepts and at the reject rate on the
    rest.  The rates are unknown, and ``min_signal_strength`` bounds both away
    from the boundary -- a bound only makes the test reject less.

    A cut that mixes the classes depletes the accept side and enriches the reject
    side, so the tail is fixed by the side and never by where its rate falls.
    """
    eps = pst.table.column(seed_row)[pst.table.representative]
    signal = pst.config.min_signal_strength
    worst = 1.0
    for side, rate, depletes in (
        (decision >= pst.accept_thresh, decision_boundary + signal, True),
        (decision < pst.reject_thresh, decision_boundary - signal, False),
    ):
        n = int(side.sum())
        # A bound outside [0, 1] carries no evidence, so the side cannot reject.
        if n < 2 or not 0 < rate < 1:
            continue
        hits = int(eps[side].sum())
        tail = (
            scipy.stats.binom.cdf(hits, n, rate)
            if depletes
            else scipy.stats.binom.sf(hits - 1, n, rate)
        )
        worst = min(worst, float(tail))
    return min(1.0, 2 * worst)


class AcceptPreservingGate:
    """Holds each suffix family to the accept-preserving split, across the loop
    that resamples until one passes.  Carries the give-up budget, which is spent
    only on families that were actually tested."""

    def __init__(self, config):
        self.enabled = config.require_accept_preserving
        self.rejections = 0

    def admits(self, pst, decision, decision_boundary, seed_row) -> bool:
        if not self.enabled:
            return True
        p = accept_preserving_pvalue(pst, decision, decision_boundary, seed_row)
        if p >= ACCEPT_PRESERVING_ALPHA:
            self.rejections = 0
            return True
        self.rejections += 1
        if self.rejections >= ACCEPT_PRESERVING_GIVE_UP:
            raise NoAcceptPreservingFamily(
                f"{self.rejections} families running were not accept-preserving "
                f"(last p={p:.2e}); no suffix family realises the "
                f"accept-preserving split on this target"
            )
        return False


def sample_suffix_family(pst, v: int) -> Tuple[List[int], float]:
    prev_effective_fnr = 1.0
    strategy = "suffix"
    decision_boundary = pst.decision_boundary
    family_size = smallest_readable_family(
        pst.config.min_signal_strength, decision_boundary
    )
    gate = AcceptPreservingGate(pst.config)

    while True:
        # Promotes the seed to fully observed, which identify_cluster_around
        # requires of it. Redone each round, since more prefixes may have
        # arrived since the last one.
        pst.table.column(v)
        requested = family_size
        vs, decision_boundary = identify_cluster_around(
            pst, v, requested, decision_boundary
        )
        pst.decision_boundary = decision_boundary
        clustered = len(vs)
        family_size = smallest_readable_family(
            pst.config.min_signal_strength, decision_boundary
        )

        # An undersized family is unusable whatever its FNR would measure, and
        # testing it would spend a budget that means no accept-preserving family
        # exists.  Short of what was asked for, or of what this boundary now needs.
        if clustered < requested or clustered < family_size:
            effective_fnr, reason = 1.0, "undersized"
        else:
            # Both rates are properties of the population the test runs over, so
            # read the family at a size calibrated for it.
            size, pst.evidence_margin = readable_size_and_margin(
                pst.config.min_signal_strength,
                decision_boundary,
                clustered,
                family_size,
            )
            vs = vs[:size]
            decision = pst.compute_decision(vs, pst.table.representative)
            fnr = pst.fnr_from_decision(decision)
            effective_fnr, reason = fnr, f"FNR {fnr:.4f} too high"
            if not gate.admits(pst, decision, decision_boundary, v):
                effective_fnr, reason = 1.0, "not accept-preserving"

        if effective_fnr <= pst.config.fnr_limit:
            print(
                f"FNR limit reached, decision boundary: {decision_boundary:.4f}, "
                f"margin: {pst.evidence_margin:.4f}"
            )
            return vs, decision_boundary

        if effective_fnr >= prev_effective_fnr or strategy == "prefix":
            strategy = "prefix" if strategy == "suffix" else "suffix"

        prev_effective_fnr = effective_fnr

        print(
            f"{reason}, sampling more {strategy}es; "
            f"decision_boundary: {decision_boundary:.4f}"
        )

        if strategy == "suffix":
            kept = pst.sample_more_suffixes(amount=family_size, reference=v)
            print(f"  kept {kept}/{family_size} after screening")
        else:
            pst.sample_more_prefixes()
