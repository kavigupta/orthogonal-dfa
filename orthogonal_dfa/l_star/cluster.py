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

#: Confidence the drift bounds below are read at.  A bound that is loose only
#: costs prefixes, and the interval narrows as the square root of them, so this
#: buys a good deal of evidence for the strength it gives up.
ACCEPT_PRESERVING_CONFIDENCE = 0.05

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


def _share(displacement: float, signal: float) -> float:
    """The displacement of a side's rate, as the share of it that is the other
    class: a share of ``f`` moves the rate by ``2 * signal * f``."""
    return min(1.0, max(0.0, displacement / (2 * signal)))


def _rate_interval(hits: int, n: int) -> Tuple[float, float]:
    """Clopper-Pearson interval for ``hits``/``n``."""
    conf = ACCEPT_PRESERVING_CONFIDENCE
    low = scipy.stats.beta.ppf(conf, hits, n - hits + 1) if hits else 0.0
    high = scipy.stats.beta.ppf(1 - conf, hits + 1, n - hits) if hits < n else 1.0
    return float(low), float(high)


def certification_sample(pst, vs, amount: int):
    """Prefixes drawn only to read the split on, and never added to the table.

    Reading one costs a query per family member, plus the one for the split
    itself.  Adding it to the table instead costs a query per fully observed
    column -- an order of magnitude more once the pool has grown -- and it
    unsettles the FNR the round has only just met, which is bought back with a
    fresh cohort of suffixes that every later prefix is then read against.
    """
    prefixes = [
        pst.sampler.sample(pst.rng, alphabet_size=pst.alphabet_size)
        for _ in range(amount)
    ]
    suffixes = [pst.table.suffix(v) for v in vs]
    pairs = [list(p) + list(sfx) for p in prefixes for sfx in suffixes]
    read = pst.table.memo.membership_queries(pairs + [list(p) for p in prefixes])
    family = np.asarray(read[: len(pairs)]).reshape(len(prefixes), len(suffixes))
    return family.mean(1), np.asarray(read[len(pairs) :])


def _split_counts(pst, decision, seed_row, extra):
    """``((hits, n), (hits, n))`` for the accept and reject sides of the cut,
    counted on the split's own column, or ``None`` if a side is empty."""
    column = pst.table.column(seed_row)[pst.table.representative]
    if extra is not None:
        extra_decision, extra_column = extra
        decision = np.concatenate([decision, extra_decision])
        column = np.concatenate([column, extra_column])
    counts = []
    for side in (decision >= pst.accept_thresh, decision < pst.reject_thresh):
        n = int(side.sum())
        # One side on its own says nothing about a gap.
        if n == 0:
            return None
        counts.append((int(column[side].sum()), n))
    return tuple(counts)


def _gap_interval(counts, signal: float) -> Tuple[float, float]:
    """Interval on the share of the cut that is the other class, from the gap the
    split's column reads across it."""
    rates = [(h / n,) + _rate_interval(h, n) for h, n in counts]
    (acc, acc_low, acc_high), (rej, rej_low, rej_high) = rates
    # Newcombe: each side carries its own error into the difference.
    gap = acc - rej
    below = ((acc - acc_low) ** 2 + (rej_high - rej) ** 2) ** 0.5
    above = ((acc_high - acc) ** 2 + (rej - rej_low) ** 2) ** 0.5
    return (
        _share(2 * signal - (gap + above), signal),
        _share(2 * signal - (gap - below), signal),
    )


def accept_preserving_drift(pst, decision, seed_row, extra=None) -> Tuple[float, float]:
    """Interval on the share of the family's cut that is the other class.

    Membership of ``p + v`` is membership of ``p`` for the empty suffix, so its
    column is the accept-preserving split itself: a family that realises the
    split reads it ``2 * signal`` apart across its own cut, and mixing the
    classes closes that gap in proportion to the share mixed in.

    Read as the gap, rather than as two rates about ``decision_boundary``.  That
    boundary is estimated from the family's decision, a mean over all its
    suffixes, while this is a single column of one; whatever displaces the two
    together belongs to that difference and not to the split, and a gap does not
    see it where a pair of rates does.
    """
    counts = _split_counts(pst, decision, seed_row, extra)
    if counts is None:
        return 0.0, 1.0
    return _gap_interval(counts, pst.config.min_signal_strength)


def prefixes_to_certify(pst, decision, seed_row, vs, tolerated: float) -> int:
    """Prefixes to draw for the split alone, enough that the gap it reads closes
    on ``tolerated`` at the rates the evidence in hand already shows.

    Grown against the interval itself rather than a normal's square root: it is a
    Clopper-Pearson interval, and at these counts the two do not agree.

    Capped where the draw costs what the round it stands in for would have.
    Growing the pool spends a query on every fully observed column; a prefix read
    only for the split spends one per family member and one for the split, so
    parity between them is the ratio.
    """
    held = len(pst.table.prefixes)
    counts = _split_counts(pst, decision, seed_row, None)
    columns = max(1, len(pst.table.fully_observed()))
    parity = pst.config.num_addtl_prefixes * columns // (len(vs) + 1)
    if counts is None:
        return parity
    signal = pst.config.min_signal_strength
    # Scale the counts as drawing more of the same would, and stop where the
    # interval clears; beyond parity the pool round was the cheaper way to ask.
    for scale in range(2, 2 + parity // max(held, 1)):
        grown = tuple((h * scale, n * scale) for h, n in counts)
        if _gap_interval(grown, signal)[1] <= tolerated:
            return held * (scale - 1)
    return parity


class AcceptPreservingGate:
    """Holds each suffix family to the accept-preserving split, across the loop
    that resamples until one passes.  Spends the give-up budget only on families
    shown to have drifted, never on ones there was not the evidence to judge."""

    def __init__(self, config):
        self.enabled = config.require_accept_preserving
        self.rejections = 0
        self.uncertified = 0

    def tolerated(self, pst) -> float:
        return tolerated_drift(pst.config.min_signal_strength, pst.evidence_margin)

    def verdict(self, pst, decision, seed_row, extra=None) -> str:
        if not self.enabled:
            return ADMITTED
        low, high = accept_preserving_drift(pst, decision, seed_row, extra)
        tolerated = self.tolerated(pst)
        if high <= tolerated:
            self.rejections = self.uncertified = 0
            return ADMITTED
        # Drift this far is consistent with the evidence but so is none of it.
        if low <= tolerated:
            self.uncertified += 1
            # The interval narrows with the prefixes, so a family still straddling
            # the tolerance after this many rounds of them is sitting on it, and
            # more will not settle what it is.
            if self.uncertified >= ACCEPT_PRESERVING_GIVE_UP:
                raise NoAcceptPreservingFamily(
                    f"{self.uncertified} rounds of prefixes left the family's "
                    f"drift somewhere in [{low:.0%}, {high:.0%}], astride the "
                    f"{tolerated:.0%} the decision can absorb; no suffix family "
                    f"realises the accept-preserving split on this target"
                )
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
        elif fnr <= pst.config.fnr_limit:
            # Only a family the round would otherwise return is worth certifying.
            # One still failing the FNR is being resampled whatever the split
            # looks like, and asking anyway spends the budget for a verdict on a
            # family that never had to have one.
            verdict = gate.verdict(pst, decision, v)
            if verdict is UNCERTIFIED:
                verdict = gate.verdict(
                    pst,
                    decision,
                    v,
                    certification_sample(
                        pst,
                        vs,
                        prefixes_to_certify(pst, decision, v, vs, gate.tolerated(pst)),
                    ),
                )
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
