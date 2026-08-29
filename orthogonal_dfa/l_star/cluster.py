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

#: Chance of calling a family drifted when it is not, or clean when it is not.
ACCEPT_PRESERVING_ERROR_RATE = 0.05

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

    Per side, but the gap the drift is read from closes by the two sides added
    together and cannot say which of them it came off.  So the sum is held to
    this, which is what it takes for either side alone to be: the other could be
    carrying none of it.  Two sides carrying half of it each are refused for the
    same reason -- from the gap they read as one side carrying all of it.
    """
    return (min_signal_strength - evidence_margin) / (2 * min_signal_strength)


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


def _split_counts(pst, decision, seed_row, extra=None):
    """``((hits, n), (hits, n))`` for the accept and reject sides of the cut,
    counted on the split's own column.

    Both sides carry prefixes: a family that leaves one empty has an FNR of 1,
    and is resampled without ever reaching the gate.
    """
    column = pst.table.column(seed_row)[pst.table.representative]
    if extra is not None:
        extra_decision, extra_column = extra
        decision = np.concatenate([decision, extra_decision])
        column = np.concatenate([column, extra_column])
    counts = []
    for side in (decision >= pst.accept_thresh, decision < pst.reject_thresh):
        n = int(side.sum())
        assert n, "a gap needs both sides of the cut"
        counts.append((int(column[side].sum()), n))
    return tuple(counts)


def observed_drift(counts, signal: float) -> float:
    """The two sides' shares of the other class added together, as the counts
    read it.  The gap closes by ``2 * signal`` times that sum, and cannot say
    which side it came off."""
    (hits_a, n_a), (hits_r, n_r) = counts
    return _clamp(1 - (hits_a / n_a - hits_r / n_r) / (2 * signal))


def _gap_pvalue(counts, gap: float, wider: bool, reject_rate_grid_size=201) -> float:
    """Chance of a gap at least (or at most) the observed one, were the true gap
    exactly ``gap``.

    Exact throughout: the accept count is summed over its binomial and each term
    weighted by the reject side's own binomial tail.  Where the reject side's
    rate itself sits is unknown, and the answer has to hold wherever that is, so
    it is read at the rate that makes the gap hardest to call and not at an
    estimate.  A difference of two binomials is not normal at these counts and is
    not treated as one.
    """
    (hits_a, n_a), (hits_r, n_r) = counts
    observed = hits_a / n_a - hits_r / n_r
    accepts = np.arange(n_a + 1)
    edge = n_r * (accepts / n_a - observed)
    worst = 0.0
    for rate in np.linspace(0, max(0.0, 1 - gap), reject_rate_grid_size):
        weight = scipy.stats.binom.pmf(accepts, n_a, min(1.0, rate + gap))
        reject_side = (
            scipy.stats.binom.cdf(np.floor(edge), n_r, rate)
            if wider
            else scipy.stats.binom.sf(np.ceil(edge) - 1, n_r, rate)
        )
        worst = max(worst, float((weight * reject_side).sum()))
    return worst


def _clamp(share: float) -> float:
    return min(1.0, max(0.0, share))


def drift_verdict(counts, signal: float, tolerated: float) -> str:
    """Whether the cut carries more of each class on the other's side than the
    decision can absorb, or less, or whether the counts do not say."""
    # The gap a family drifted exactly to the tolerance would read.
    gap = 2 * signal * (1 - tolerated)
    if _gap_pvalue(counts, gap, wider=True) <= ACCEPT_PRESERVING_ERROR_RATE:
        return ADMITTED
    if _gap_pvalue(counts, gap, wider=False) <= ACCEPT_PRESERVING_ERROR_RATE:
        return DRIFTED
    return UNCERTIFIED


def prefixes_to_certify(pst, counts, vs, tolerated: float) -> int:
    """How many prefixes to draw for the split alone, to settle a verdict the
    prefixes in hand left undecided.

    How many it takes depends on the rates, so the rates in hand are the guess:
    if the same ones held over twice the counts, or three times, would the
    verdict come out decided?  The first multiple that would is the answer.

    Drawn against the representative prefixes, which are what the sampler
    returns, so a draw of that size again brings a side of that size again --
    the core is not drawn and does not count toward it.

    Never more than the round of pooled prefixes this stands in for would have
    cost.  One of those spends a query on every fully observed column, where one
    read for the split spends a query per family member and one for the split
    itself, so the budget in prefixes is the ratio between them.
    """
    drawn = max(1, int(pst.table.representative.sum()))
    columns = max(1, len(pst.table.fully_observed()))
    budget = pst.config.num_addtl_prefixes * columns // (len(vs) + 1)
    signal = pst.config.min_signal_strength
    for multiple in range(2, 2 + budget // drawn):
        supposed = tuple((hits * multiple, n * multiple) for hits, n in counts)
        if drift_verdict(supposed, signal, tolerated) is not UNCERTIFIED:
            return drawn * (multiple - 1)
    return budget


class AcceptPreservingGate:
    """Holds each suffix family to the accept-preserving split, across the loop
    that resamples until one passes.  Carries the give-up budget, spent on every
    round that does not produce a family the split can be certified on.

    Nothing resets that budget: admitting a family is the round returning, and
    the gate is made afresh for the next search."""

    def __init__(self, config):
        self.enabled = config.require_accept_preserving
        self.refusals = 0

    def verdict(self, pst, decision, seed_row, vs) -> str:
        if not self.enabled:
            return ADMITTED
        signal = pst.config.min_signal_strength
        tolerated = tolerated_drift(signal, pst.evidence_margin)
        counts = _split_counts(pst, decision, seed_row)
        verdict = drift_verdict(counts, signal, tolerated)
        if verdict is UNCERTIFIED:
            # Undecided on what the pool holds, and the pool is the dear way to
            # ask: draw prefixes for the split alone and read it again.
            wanted = prefixes_to_certify(pst, counts, vs, tolerated)
            counts = _split_counts(
                pst, decision, seed_row, certification_sample(pst, vs, wanted)
            )
            verdict = drift_verdict(counts, signal, tolerated)
        if verdict is ADMITTED:
            return ADMITTED
        self.refusals += 1
        if self.refusals >= ACCEPT_PRESERVING_GIVE_UP:
            drift = observed_drift(counts, signal)
            settled = "past" if verdict is DRIFTED else "neither inside nor past"
            raise NoAcceptPreservingFamily(
                f"{self.refusals} families running put {drift:.0%} of each class "
                f"on the other's side between them, {settled} the {tolerated:.0%} "
                f"a side may carry; no suffix family realises the "
                f"accept-preserving split on this target"
            )
        return verdict


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
            # Certify only right before returning, as certifying is expensive.
            verdict = gate.verdict(pst, decision, v, vs)
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
            # Suffixes are clustered by how they read across the prefixes, so
            # sampling more of them with the same prefixes picks a family much
            # like this one. Only more prefixes change which suffixes group.
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
