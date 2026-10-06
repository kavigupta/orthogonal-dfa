import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .certificate import clopper_pearson
from .mask_table import UNIFORM
from .prefix_populations import grow_population, population_labels, prefixes_for_split
from .statistics import (
    cross_limit_for_coverage_error,
    evidence_margin_for_population_size,
    population_size_and_evidence_margin,
)
from .suffix_groups import nearest_to_anchor_group


def identify_cluster_around(
    pst, seed: int, count: int, decision_boundary: float
) -> Tuple[List[int], float]:
    """Seed, then the first count - 1 of the rest of the pool in
    nearest_to_anchor_group's order, anchored on seed's reads."""
    # Restrict to representative prefix columns: the suffix family and the
    # decision boundary are global calibration, and a caller that has re-scoped
    # them means that scope to be what calibration reads.
    candidate = np.array(pst.suffix_pool)
    reads = pst.table.observed_masks(candidate, pst.table.representative)
    reads = reads.astype(float)
    assert seed in pst.suffix_pool, "cluster seed must be in the pool"
    seed_local = pst.suffix_pool.index(seed)
    signal = pst.config.min_signal_strength
    others = np.flatnonzero(np.arange(len(reads)) != seed_local)
    cluster = [seed_local]
    if len(others):
        order = nearest_to_anchor_group(
            reads[others],
            reads[seed_local],
            list(pst.table.population_masks().values()),
            pst.table.seed_scoring(),
            (decision_boundary - signal, decision_boundary + signal),
            # As many clusters as a group the preconditions guarantee, a
            # min_suffix_frequency share of the pool, needs to get one of its own.
            k=math.ceil(1 / pst.config.min_suffix_frequency),
            alpha=pst.config.screening_alpha,
            rng=pst.rng,
        )
        cluster += others[order[: count - 1]].tolist()

    # Estimate decision boundary from the prefix separation
    prefix_means = reads[cluster].mean(0)
    cluster_center = prefix_means > decision_boundary
    accept_prefixes = prefix_means[cluster_center]
    reject_prefixes = prefix_means[~cluster_center]
    # A one-sided cluster has only the one class's mean to go on, which sits a
    # signal away from the boundary.  Reading the boundary off it directly would
    # cut that class down the middle, so step off it by the signal we were promised.
    if len(accept_prefixes) > 0 and len(reject_prefixes) > 0:
        decision_boundary = (accept_prefixes.mean() + reject_prefixes.mean()) / 2
    elif len(accept_prefixes) > 0:
        decision_boundary = accept_prefixes.mean() - signal
    elif len(reject_prefixes) > 0:
        decision_boundary = reject_prefixes.mean() + signal

    # Keep the implied rates, boundary +/- the signal, probabilities.
    decision_boundary = min(max(decision_boundary, signal), 1 - signal)

    return candidate[cluster].tolist(), decision_boundary


def round_rates(pst):
    """
    (p_0, p_1), the current round's assumption for
        P[Oracle = 1 | Noiseless oracle = i]
    for i in (0, 1)
    """
    signal = pst.config.min_signal_strength
    return pst.decision_boundary - signal, pst.decision_boundary + signal


def read_rates(pst, decision_boundary):
    """The rates a family is read at.

    A caller may ask for any crispness it likes; how far the split then sits from
    accept preservation is bounded by the band those rates buy, so the rate asked
    for is used only where it already holds ``max_coverage_error``.
    """
    return (
        min(
            pst.config.cross_limit,
            cross_limit_for_coverage_error(
                pst.config.min_signal_strength,
                pst.acceptable_fnr,
                pst.config.max_coverage_error,
                center=decision_boundary,
            ),
        ),
        pst.acceptable_fnr,
    )


def smallest_readable_family(min_signal_strength, decision_boundary, rates):
    """Fewest suffixes a decision at this boundary can be read over.

    How many it needs depends on where the boundary sits: the two classes draw
    from binomials whose variance differs once it leaves 0.5.
    """
    cross_limit, acceptable_fnr = rates
    size, _ = population_size_and_evidence_margin(
        min_signal_strength, cross_limit, acceptable_fnr, center=decision_boundary
    )
    return size


def readable_size_and_margin(
    min_signal_strength, decision_boundary, have, smallest, rates
):
    """The largest size at or below ``have`` whose band holds both error rates, and
    the margin that reads it.  ``have`` must be at least ``smallest``.

    Sizes just above the minimum can admit no band at all -- one more suffix shifts
    every operating point off the integer lattice -- so step down rather than call
    a family that is large enough undersized. ``smallest`` always admits one, so
    the walk cannot run off the end.
    """
    cross_limit, acceptable_fnr = rates
    for size in range(have, smallest - 1, -1):
        found = evidence_margin_for_population_size(
            min_signal_strength,
            cross_limit,
            acceptable_fnr,
            size,
            center=decision_boundary,
        )
        if found is not None:
            return size, found[1]
    raise AssertionError(f"{smallest} suffixes was supposed to be readable")


#: Rejections in a row after which no accept-preserving family is believed to
#: exist.  More suffixes is the only remedy, and none help against a target where
#: no suffix preserves the accept/reject classes.
ACCEPT_PRESERVING_GIVE_UP = 20

#: P(admitting a family with E_j > misclassification_limit for some population j)
#: over both of the gate's drift_verdict calls on it, E_j as in misclassified_bounds.
ACCEPT_PRESERVING_ERROR_RATE = 0.05
#: Each drift_verdict call's share of it.
ACCEPT_PRESERVING_VERDICT_ERROR_RATE = ACCEPT_PRESERVING_ERROR_RATE / 2

#: What the gate was able to conclude about a family.
ADMITTED, DRIFTED, UNCERTIFIED = "admitted", "drifted", "uncertified"


class NoAcceptPreservingFamily(Exception):
    """No accept-preserving suffix family could be sampled for this target."""


def certification_sample(pst, vs, by_population):
    """``label -> (family means, split column)`` for prefixes read only to
    settle the split, and never added to the table.

    Reading one costs a query per family member, plus the one for the split
    itself.  Adding it to the table instead costs a query per pooled
    suffix -- an order of magnitude more once the pool has grown -- and it
    unsettles the FNR the round has only just met, which is bought back with a
    fresh cohort of suffixes that every later prefix is then read against.
    """
    suffixes = [pst.table.suffix(v) for v in vs]
    out = {}
    for label, prefixes in by_population.items():
        pairs = [p + sfx for p in prefixes for sfx in suffixes]
        read = pst.table.memo.membership_queries(pairs + prefixes)
        family = np.asarray(read[: len(pairs)]).reshape(len(prefixes), len(suffixes))
        out[label] = (family.mean(1), np.asarray(read[len(pairs) :]))
    return out


def _split_counts(pst, reads):
    """
    The family's classification of the prefixes when run in "decisive mode" where
        there is no indecision band; jointly distributed with the oracle's labels.

    Returns label -> ((a_1, n_1), (a_0, n_0)).

        n_i = # of prefixes the family places in class i, decisively
        a_i = # of prefixes the family places in class i that the oracle assigns a 1 to
    """
    return {
        label: tuple(
            (int(column[side].sum()), int(side.sum()))
            for side in (
                decision >= pst.decision_boundary,
                decision < pst.decision_boundary,
            )
        )
        for label, (decision, column) in reads.items()
    }


def misclassified_bounds(pst, by_population, level):
    """
    Returns label j -> (bound_j, point_j) where for

        E_j = P_{p ~ D_j}[ decisiveClassifyFamily(p) != noiselessOracle(p) ],

    point_j is E_j assuming that the fractions a_i/n_i and n_i / (n_0 + n_1)
    arising from _split_counts are exact expectations rather than samples, and
    bound_j satisfies

        P(E_j <= bound_j for every j) >= 1 - level

    under only the condition that the oracle has exactly round_rates(pst) rates
    (in particular, it does not assume the aforementioned fractions are exact
    expectations).
    """
    drawn = {
        label: counts
        for label, counts in by_population.items()
        if counts[0][1] + counts[1][1]
    }
    each = level / (4 * max(1, len(drawn)))
    p_0, p_1 = round_rates(pst)

    def wrong(accept, reject):
        """For the accepting and rejecting sides reading 1 at rates accept and
        reject, the share e of each in the other class, clipped to [0, 1], from

            accept = p_1 - e (p_1 - p_0),   reject = p_0 + e (p_1 - p_0)."""
        return np.clip(
            [(p_1 - accept) / (p_1 - p_0), (reject - p_0) / (p_1 - p_0)], 0, 1
        )

    out = {}
    for label, ((hits_a, n_a), (hits_r, n_r)) in drawn.items():
        sides = np.array([n_a, n_r])
        ones = np.array([hits_a, hits_r])
        mass_low, mass_high = clopper_pearson(sides, np.full(2, sides.sum()), each)
        low, high = clopper_pearson(ones, sides, each)
        worst = wrong(low[0], high[1])
        # The total is linear in the accepting side's share, so it is largest at
        # an end of what both sides' intervals allow that share.
        ends = (max(mass_low[0], 1 - mass_high[1]), min(mass_high[0], 1 - mass_low[1]))
        read = ones / np.maximum(sides, 1)
        out[label] = (
            float(max(a * worst[0] + (1 - a) * worst[1] for a in ends)),
            float(sides / sides.sum() @ wrong(read[0], read[1])),
        )
    return out


def misclassification_limit(pst, label) -> float:
    return pst.config.max_coverage_error if label == UNIFORM else 1 / 2


def _population_verdict(pst, label, bound, point):
    """ADMITTED if bound <= limit, else DRIFTED if point > limit, else
    UNCERTIFIED, for limit = misclassification_limit."""
    limit = misclassification_limit(pst, label)
    if bound <= limit:
        return ADMITTED
    if point > limit:
        return DRIFTED
    return UNCERTIFIED


def drift_verdict(pst, by_population, level):
    """(_population_verdict of j, j) for j = argmax_j (bound_j - limit_j), with
    (bound_j, point_j) from misclassified_bounds; None for j when ADMITTED."""
    bounds = misclassified_bounds(pst, by_population, level)
    if not bounds:
        return UNCERTIFIED, UNIFORM
    worst = max(
        bounds, key=lambda label: bounds[label][0] - misclassification_limit(pst, label)
    )
    verdict = _population_verdict(pst, worst, *bounds[worst])
    return verdict, None if verdict is ADMITTED else worst


def alignment_size(pst, populations, limit) -> int:
    """
    Fewest prefixes, a power of 2, on which a population with limit `limit`
    would get _population_verdict = ADMITTED, for a family that satisfies the
    below conditions
        - perfectly classifies all populations, with half the prefixes on each
          side of its cut
        - every observed fraction equal to its expectation
    `populations` is how many populations share the error rate
    """

    p_0, p_1 = round_rates(pst)
    size = 2
    while True:
        half = size // 2
        counts = ((round(p_1 * half), half), (round(p_0 * half), half))
        bound, _ = misclassified_bounds(
            pst,
            dict.fromkeys(range(populations), counts),
            ACCEPT_PRESERVING_VERDICT_ERROR_RATE,
        )[0]
        if bound <= limit:
            return size
        size *= 2


def certification_budget(pst, vs) -> int:
    """Never more prefixes than the round of pooled prefixes this stands in for
    would have cost.  One of those spends a query on every pooled suffix, where
    one read for the split spends a query per family member and one for the
    split itself, so the budget in prefixes is the ratio between them.
    """
    columns = max(1, len(pst.suffix_pool))
    return max(1, pst.config.num_addtl_prefixes * columns // (len(vs) + 1))


def prefixes_to_certify(pst, counts, label, level, vs) -> int:
    """n (m - 1), n label's prefixes drawn, for the smallest m >= 2 at which
    label's counts times m are not UNCERTIFIED by _population_verdict; at most
    certification_budget."""
    budget = certification_budget(pst, vs)
    (a_1, n_1), (a_0, n_0) = counts.get(label, ((0, 0), (0, 0)))
    drawn = n_1 + n_0
    if not drawn:
        return budget
    for m in range(2, 2 + budget // drawn):
        scaled = {**counts, label: ((m * a_1, m * n_1), (m * a_0, m * n_0))}
        bound, point = misclassified_bounds(pst, scaled, level)[label]
        if _population_verdict(pst, label, bound, point) is not UNCERTIFIED:
            return drawn * (m - 1)
    return budget


class AcceptPreservingGate:
    """Holds each suffix family to the accept-preserving split, across the loop
    that resamples until one passes.  Carries the give-up budget, spent on every
    round that does not produce a family the split can be certified on.

    Nothing resets that budget: admitting a family is the round returning, and
    the gate is made afresh for the next search."""

    def __init__(self, config, state):
        self.enabled = config.require_accept_preserving
        self.refusals = 0
        self._state = state
        self._drawn = None
        self._sizes = None

    def _certification_prefixes(self, pst, voters):
        """``label -> prefixes`` to certify a family over, drawn for the round
        and read by every family it tries."""
        if self._drawn is None:
            labels = population_labels(self._state)
            budget = certification_budget(pst, voters)
            self._sizes = {
                label: min(
                    alignment_size(
                        pst, len(labels), misclassification_limit(pst, label)
                    ),
                    budget,
                )
                for label in labels
            }
            drawn = {
                label: prefixes_for_split(pst, self._state, label, self._sizes[label])
                for label in labels
            }
            self._drawn = {label: held for label, held in drawn.items() if held}
        return self._drawn

    def _certify_further(self, pst, counts, label, voters):
        """counts with a further read of label's population added into it."""
        held = self._drawn.get(label, [])
        more = prefixes_for_split(
            pst,
            self._state,
            label,
            prefixes_to_certify(
                pst, counts, label, ACCEPT_PRESERVING_VERDICT_ERROR_RATE, voters
            ),
        )
        if not more:
            return counts
        # Kept, so a later family is read on these rather than buying them again.
        self._drawn[label] = held + more
        extra = _split_counts(pst, certification_sample(pst, voters, {label: more}))
        empty = ((0, 0), (0, 0))
        return {
            **counts,
            label: tuple(
                (hits + grown_hits, n + grown_n)
                for (hits, n), (grown_hits, grown_n) in zip(
                    counts.get(label, empty), extra.get(label, empty)
                )
            ),
        }

    def verdict(self, pst, seed_row, vs):
        """``(verdict, label)``: what the split says, and which population said
        it, for the search to answer."""
        if not self.enabled:
            return ADMITTED, None
        # The family was clustered over the table's prefixes, and a large enough
        # pool fits their noise, so only prefixes it never saw can test it.  The
        # seed votes on p with the very read of p being scored, so it sits out.
        voters = [u for u in vs if u != seed_row]
        prefixes = self._certification_prefixes(pst, voters)
        counts = _split_counts(pst, certification_sample(pst, voters, prefixes))
        verdict, blamed = drift_verdict(
            pst, counts, ACCEPT_PRESERVING_VERDICT_ERROR_RATE
        )
        if verdict is UNCERTIFIED:
            counts = self._certify_further(pst, counts, blamed, voters)
            verdict, blamed = drift_verdict(
                pst, counts, ACCEPT_PRESERVING_VERDICT_ERROR_RATE
            )
        if verdict is ADMITTED:
            return ADMITTED, None
        if verdict is DRIFTED:
            # A refusal scores the population's own reads, which every later family
            # would read again: kept, one unlucky sample refuses each sound family
            # in turn until the search gives up.
            redrawn = prefixes_for_split(pst, self._state, blamed, self._sizes[blamed])
            if redrawn:
                self._drawn[blamed] = redrawn
            else:
                self._drawn.pop(blamed, None)
        self.refusals += 1
        if self.refusals >= ACCEPT_PRESERVING_GIVE_UP:
            bounds = misclassified_bounds(
                pst, counts, ACCEPT_PRESERVING_VERDICT_ERROR_RATE
            )
            if blamed in bounds:
                bound, point = bounds[blamed]
                read = (
                    f"misclassifies {point:.0%} of {blamed} at the rates read, "
                    f"at most {bound:.0%}, against "
                    f"{misclassification_limit(pst, blamed):.0%}"
                )
            else:
                read = f"drew no prefix to read {blamed} on"
            raise NoAcceptPreservingFamily(
                f"{self.refusals} families refused: the last {read}; no suffix "
                f"family realises the accept-preserving split on this target"
            )
        return verdict, blamed


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
    #: The population the search grows to answer this: the FNR's argmax, or the
    #: one the gate refused on.
    blamed: Optional[object]


def judge_family(pst, gate, v, vs, family_size) -> Judged:
    """Read the clustered family, and say what stands against using it.

    Sets the margin the family is read with, which the caller reports.
    """
    # An undersized family is unusable whatever its FNR would measure, and
    # testing it would spend a budget that means no accept-preserving family
    # exists.
    if len(vs) < family_size:
        return Judged(vs, 1.0, "undersized", ADMITTED, None)
    # Both rates are properties of the population the test runs over, so read
    # the family at a size calibrated for it.
    size, pst.evidence_margin = readable_size_and_margin(
        pst.config.min_signal_strength,
        pst.decision_boundary,
        len(vs),
        family_size,
        read_rates(pst, pst.decision_boundary),
    )
    vs = vs[:size]
    decision = pst.compute_decision(vs, pst.table.representative)
    fnr, worst = pst.fnr_from_decision(decision)
    too_high = f"FNR {fnr:.4f} too high"
    if fnr > pst.fnr_limit:
        return Judged(vs, fnr, too_high, ADMITTED, worst)
    # Certify only right before returning, as certifying is expensive.
    verdict, blamed = gate.verdict(pst, v, vs)
    if verdict is DRIFTED:
        return Judged(vs, 1.0, "not accept-preserving", verdict, blamed)
    if verdict is UNCERTIFIED:
        return Judged(vs, 1.0, "accept-preserving not established", verdict, blamed)
    return Judged(vs, fnr, too_high, verdict, worst)


def sample_suffix_family(pst, v: int, state) -> Tuple[List[int], float]:
    """A suffix family clustered around ``v``, held to the accept-preserving
    split before it is returned.

    ``v`` is the empty suffix from either caller, and the gate reads the split
    off its column on the strength of that: membership of ``p + v`` is
    membership of ``p`` only while ``v`` is empty.

    ``state`` carries the round's prefix populations, which a refusal grows.
    """
    prev_effective_fnr = 1.0
    strategy = "suffix"
    decision_boundary = pst.decision_boundary
    family_size = smallest_readable_family(
        pst.config.min_signal_strength,
        decision_boundary,
        read_rates(pst, decision_boundary),
    )
    gate = AcceptPreservingGate(pst.config, state)

    if v not in pst.suffix_pool:
        pst.suffix_pool.append(v)
    while True:
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
                pst.config.min_signal_strength,
                decision_boundary,
                read_rates(pst, decision_boundary),
            )
            if len(vs) >= family_size:
                break

        judged = judge_family(pst, gate, v, vs, family_size)

        if judged.fnr <= pst.fnr_limit:
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
            kept, drawn = pst.sample_more_suffixes(amount=family_size, reference=v)
            print(f"  wanted {family_size} more suffixes, kept {kept} of {drawn} drawn")
        elif judged.blamed is None:
            pst.sample_more_prefixes()
        elif not grow_population(pst, state, judged.blamed):
            # Fallback, this should very rarely happen. At this point, the
            # algorithm has detected a precondition violation, so later
            # results do not follow the theory.

            # Likely precondition violation is too-high collision probability
            # among suffixes.
            kept, drawn = pst.sample_more_suffixes(amount=family_size, reference=v)
            print(f"  nothing draws for {judged.blamed}; kept {kept} of {drawn}")
            strategy = "suffix"
