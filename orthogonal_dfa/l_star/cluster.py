from dataclasses import dataclass
from typing import List, Optional, Tuple

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
    # A prefix is decisive or not according to the suffixes picked here, so a
    # population with no say in the picking is one the family is not chosen to
    # separate -- and the FNR is then read over it population by population.
    # Weighted so each contributes the same however many prefixes it holds.
    weights = np.zeros(masks.shape[1])
    for population in pst.table.strata_masks().values():
        if population.any():
            weights[population] = 1 / population.sum()
    # Only keep clustering while the seed belongs to the cluster.
    # We want to avoid drifting the cluster center away from the seed, which can
    # happen if the seed has a very small cluster relative to `count`.
    cluster = [seed_local]
    loss = float("inf")
    while True:
        cluster_center = masks[cluster].mean(0) > decision_boundary
        losses = ((masks != cluster_center) * weights).sum(1)
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

#: Chance of calling a family drifted when it is not, or clean when it is not.
ACCEPT_PRESERVING_ERROR_RATE = 0.05

#: What the gate was able to conclude about a family.
ADMITTED, DRIFTED, UNCERTIFIED = "admitted", "drifted", "uncertified"


class NoAcceptPreservingFamily(Exception):
    """No accept-preserving suffix family could be sampled for this target."""


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
    pairs = [p + sfx for p in prefixes for sfx in suffixes]
    read = pst.table.memo.membership_queries(pairs + prefixes)
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
        assert n, "the split needs both sides of the cut"
        counts.append((int(column[side].sum()), n))
    return tuple(counts)


def drift_verdict(pst, counts) -> str:
    """Whether each side of the cut reads as its own class on the split, or as
    the other's, or whether the counts do not say.

    Membership of ``p + v`` is membership of ``p`` for the empty suffix, so the
    split's column says what the oracle makes of the prefixes themselves.  A
    family realises the accept-preserving split when the prefixes it calls
    accepting read there as accepting -- by the same thresholds the family is
    read with, since it is that reading being checked and not another.

    So the sides are held to ``accept_thresh`` and ``reject_thresh`` directly.
    Neither is a rate anything has to be estimated against, which is what a gap
    between the sides would have needed, and would have had to name a signal for.
    """
    hits_a, n_a = counts[0]
    hits_r, n_r = counts[1]
    alpha = ACCEPT_PRESERVING_ERROR_RATE
    # Both sides must clear their own test, so between them they cannot exceed
    # the rate either one spends.
    if (
        scipy.stats.binom.sf(hits_a - 1, n_a, pst.accept_thresh) <= alpha
        and scipy.stats.binom.cdf(hits_r, n_r, pst.reject_thresh) <= alpha
    ):
        return ADMITTED
    # Either side drifting on its own is enough to say so, so they share.
    if (
        scipy.stats.binom.cdf(hits_a, n_a, pst.accept_thresh) <= alpha / 2
        or scipy.stats.binom.sf(hits_r - 1, n_r, pst.reject_thresh) <= alpha / 2
    ):
        return DRIFTED
    return UNCERTIFIED


def prefixes_to_certify(pst, counts, vs) -> int:
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
    for multiple in range(2, 2 + budget // drawn):
        supposed = tuple((hits * multiple, n * multiple) for hits, n in counts)
        if drift_verdict(pst, supposed) is not UNCERTIFIED:
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
        counts = _split_counts(pst, decision, seed_row)
        verdict = drift_verdict(pst, counts)
        if verdict is UNCERTIFIED:
            wanted = prefixes_to_certify(pst, counts, vs)
            counts = _split_counts(
                pst, decision, seed_row, certification_sample(pst, vs, wanted)
            )
            verdict = drift_verdict(pst, counts)
        if verdict is ADMITTED:
            return ADMITTED
        self.refusals += 1
        if self.refusals >= ACCEPT_PRESERVING_GIVE_UP:
            hits_a, n_a = counts[0]
            hits_r, n_r = counts[1]
            read = "read" if verdict is DRIFTED else "could not be read"
            raise NoAcceptPreservingFamily(
                f"{self.refusals} families running {read} as cutting against the "
                f"classes: the last put {hits_a / n_a:.0%} of the prefixes it "
                f"accepts and {hits_r / n_r:.0%} of those it rejects on the "
                f"accepting side of the empty suffix, against thresholds of "
                f"{pst.accept_thresh:.0%} and {pst.reject_thresh:.0%}; no suffix "
                f"family realises the accept-preserving split on this target"
            )
        return verdict


@dataclass
class Judged:
    """A clustered family and what the round makes of it."""

    #: Read down to the size its band was calibrated for, and seeded.
    vs: List[int]
    #: What to hold against the FNR limit: what the family measures, or 1 for one
    #: that cannot be used whatever it would measure.
    fnr: float
    reason: str
    verdict: str
    #: The prefix population ``fnr`` is the rate of, and so the one to grow to
    #: answer it.  ``None`` where no population in particular is at fault.
    worst: Optional[object] = None


def judge_family(pst, gate, v, vs, family_size) -> Judged:
    """Read the clustered family, and say what stands against using it.

    Sets the margin the family is read with, which the caller reports.
    """
    # An undersized family is unusable whatever its FNR would measure, and
    # testing it would spend a budget that means no accept-preserving family
    # exists.
    if len(vs) < family_size:
        return Judged(vs, 1.0, "undersized", ADMITTED)
    # Both rates are properties of the population the test runs over, so read
    # the family at a size calibrated for it.
    size, pst.evidence_margin = readable_size_and_margin(
        pst.config.min_signal_strength,
        pst.decision_boundary,
        len(vs),
        family_size,
    )
    # By loss rank, and the seed's rank is arbitrary, so put it back: the round
    # check and the accept-preserving null are both stated about a family seeded
    # at this suffix.
    vs = vs[:size] if v in vs[:size] else [v] + vs[: size - 1]
    decision = pst.compute_decision(vs, pst.table.representative)
    fnr, worst = pst.fnr_from_decision(decision)
    too_high = f"FNR {fnr:.4f} too high"
    if fnr > pst.config.fnr_limit:
        return Judged(vs, fnr, too_high, ADMITTED, worst)
    # Certify only right before returning, as certifying is expensive.
    verdict = gate.verdict(pst, decision, v, vs)
    if verdict is DRIFTED:
        return Judged(vs, 1.0, "not accept-preserving", verdict)
    if verdict is UNCERTIFIED:
        return Judged(vs, 1.0, "accept-preserving not established", verdict)
    return Judged(vs, fnr, too_high, verdict, worst)


def sample_suffix_family(pst, v: int, grow_pool=None) -> Tuple[List[int], float]:
    """A suffix family clustered around ``v``, held to the accept-preserving
    split before it is returned.

    ``v`` is the empty suffix from either caller, and the gate reads the split
    off its column on the strength of that: membership of ``p + v`` is
    membership of ``p`` only while ``v`` is empty.

    ``grow_pool(label)`` grows one prefix population, the one the FNR is the
    rate of.  The populations are the previous round's -- that is what defines
    them -- so the caller supplies this; without one every population is
    answered by drawing uniformly, which only the uniform one answers to.
    """
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
        # The cluster is capped at the size asked for, and the boundary it
        # estimates decides the size wanted, so a boundary that moves far enough
        # leaves it short by construction.  The pool usually already holds the
        # rest: ask again at the new size before spending a cohort of oracle
        # queries on suffixes to cover a handful.
        for _ in range(2):
            vs, decision_boundary = identify_cluster_around(
                pst, v, family_size, decision_boundary
            )
            pst.decision_boundary = decision_boundary
            family_size = smallest_readable_family(
                pst.config.min_signal_strength, decision_boundary
            )
            if len(vs) >= family_size:
                break

        judged = judge_family(pst, gate, v, vs, family_size)

        if judged.fnr <= pst.config.fnr_limit:
            print(
                f"FNR limit reached, decision boundary: {decision_boundary:.4f}, "
                f"margin: {pst.evidence_margin:.4f}"
            )
            return judged.vs, decision_boundary

        if judged.verdict is UNCERTIFIED:
            # Suffixes are clustered by how they read across the prefixes, so
            # sampling more of them with the same prefixes picks a family much
            # like this one. Only more prefixes change which suffixes group.
            strategy = "prefix"
        elif judged.fnr >= prev_effective_fnr or strategy == "prefix":
            strategy = "prefix" if strategy == "suffix" else "suffix"

        prev_effective_fnr = judged.fnr

        print(
            f"{judged.reason}, sampling more {strategy}es; "
            f"decision_boundary: {decision_boundary:.4f}"
        )

        if strategy == "suffix":
            kept = pst.sample_more_suffixes(amount=family_size, reference=v)
            print(f"  kept {kept}/{family_size} after screening")
        elif grow_pool is None or judged.worst is None:
            pst.sample_more_prefixes()
        elif not grow_pool(judged.worst):
            # A population that cannot be grown is not answered by asking again.
            # What is left that moves its rate is the family read over it, so
            # buy suffixes rather than spend the round re-measuring the same
            # prefixes.
            kept = pst.sample_more_suffixes(amount=family_size, reference=v)
            print(f"  {judged.worst} is all there is; kept {kept}/{family_size}")
            strategy = "suffix"
