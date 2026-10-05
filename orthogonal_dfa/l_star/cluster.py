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
from .suffix_groups import aligned_suffixes, nearest_to_anchor_group


def identify_cluster_around(
    pst, seed: int, count: int, decision_boundary: float
) -> Tuple[List[int], float]:
    """Seed, then the first count - 1 of the rest of the pool in
    nearest_to_anchor_group's order, anchored on seed's reads."""
    # Restrict to representative prefix columns: the suffix family and the
    # decision boundary are global calibration, and a caller that has re-scoped
    # them means that scope to be what calibration reads.
    candidate = np.array(pst.suffix_pool)
    masks = pst.table.observed_masks(candidate, pst.table.representative)
    assert seed in pst.suffix_pool, "cluster seed must be in the pool"
    seed_local = pst.suffix_pool.index(seed)
    reads = masks.astype(float)
    others = np.flatnonzero(np.arange(len(reads)) != seed_local)
    cluster = [seed_local]
    if len(others):
        order = nearest_to_anchor_group(
            reads[others],
            reads[seed_local],
            list(pst.table.population_masks().values()),
            pst.table.seed_scoring(),
            (
                decision_boundary - pst.config.min_signal_strength,
                decision_boundary + pst.config.min_signal_strength,
            ),
            # As many clusters as a group the preconditions guarantee, a
            # min_suffix_frequency share of the pool, needs to get one of its own.
            k=math.ceil(1 / pst.config.min_suffix_frequency),
            alpha=pst.config.screening_alpha,
            rng=pst.rng,
        )
        cluster += others[order[: count - 1]].tolist()
    cluster_center = reads[cluster].mean(0) > decision_boundary

    # Estimate decision boundary from the prefix separation
    prefix_means = masks[cluster].mean(0)
    accept_prefixes = prefix_means[cluster_center]
    reject_prefixes = prefix_means[~cluster_center]
    signal = pst.config.min_signal_strength
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
    """(p_0, p_1) the round reads at: the boundary -/+ the signal."""
    signal = pst.config.min_signal_strength
    return pst.decision_boundary - signal, pst.decision_boundary + signal


def aligned_family(pst, seed: int, count: int) -> List[int]:
    """Seed, then the pool's aligned_suffixes against seed's reads at the
    round's rates, boundary -/+ the signal, and within max_coverage_error, read on
    the prefixes the screen never saw; count of them at most.  Every member reads each population like seed but
    for a bounded share, which is what makes some family pass the gate once the
    pool and the prefixes are large enough."""
    candidate = np.array(pst.suffix_pool)
    reads = pst.table.observed_masks(candidate, pst.table.representative)
    reads = reads.astype(float)
    seed_local = pst.suffix_pool.index(seed)
    others = np.flatnonzero(np.arange(len(reads)) != seed_local)
    scoring = pst.table.seed_scoring()
    kept = aligned_suffixes(
        reads[others],
        reads[seed_local],
        [
            m & scoring
            for m in pst.table.population_masks().values()
            if (m & scoring).any()
        ],
        round_rates(pst),
        epsilon=pst.config.max_coverage_error,
        alpha=ACCEPT_PRESERVING_ERROR_RATE,
    )
    return candidate[[seed_local] + others[kept[: count - 1]].tolist()].tolist()


def read_rates(config, decision_boundary):
    """The rates a family is read at.

    A caller may ask for any crispness it likes; how far the split then sits from
    accept preservation is bounded by the band those rates buy, so the rate asked
    for is used only where it already holds ``max_coverage_error``.
    """
    return (
        min(
            config.cross_limit,
            cross_limit_for_coverage_error(
                config.min_signal_strength,
                config.acceptable_fnr,
                config.max_coverage_error,
                center=decision_boundary,
            ),
        ),
        config.acceptable_fnr,
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

#: Chance of admitting a family that misclassifies more than max_coverage_error
#: of some population, over both of a round's looks at it.
ACCEPT_PRESERVING_ERROR_RATE = 0.05

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
    """label -> ((hits, n), (hits, n)): on the accepting and rejecting sides of the
    family's cut at the boundary, how many prefixes and how many of them the
    empty suffix reads as 1.  A population holds one class or both, so a side of
    n = 0 is ordinary."""
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
    """label -> (bound, at the rates read) on the share of the population's
    distribution the family's cut misclassifies, at the round_rates (p_0, p_1):
    a side of the cut reading r holds a share (p_1 - r) / (p_1 - p_0) of
    rejecting prefixes where it accepts and (r - p_0) / (p_1 - p_0) of accepting
    ones where it rejects, clipped to [0, 1].  The bound is the largest such
    total over Clopper-Pearson intervals at level / 4 |populations| on each
    side's share of the population and on the empty suffix's rate of 1s there,
    so every bound holds at once with probability at least 1 - level, the
    round's rates being the oracle's."""
    drawn = {
        label: counts
        for label, counts in by_population.items()
        if counts[0][1] + counts[1][1]
    }
    each = level / (4 * max(1, len(drawn)))
    p_0, p_1 = round_rates(pst)

    def wrong(accept, reject):
        """Each side's misclassified share at its rate."""
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
    """max_coverage_error for the uniform pool, drawn as the learner is scored;
    a half for every other population, which the family has only to read the
    right way round."""
    return pst.config.max_coverage_error if label == UNIFORM else 1 / 2


def drift_verdict(pst, by_population, level):
    """(verdict, label): ADMITTED when every population's misclassified_bounds
    bound is at most its misclassification_limit; otherwise, naming the
    population whose bound passes its limit furthest, DRIFTED where its share at
    the rates read is past the limit too and UNCERTIFIED where only the bound
    is.  Admitting holds the family to every population drawn, with probability
    at least 1 - level, whatever chose it."""
    bounds = misclassified_bounds(pst, by_population, level)
    if not bounds:
        return UNCERTIFIED, UNIFORM
    worst = max(
        bounds, key=lambda label: bounds[label][0] - misclassification_limit(pst, label)
    )
    bound, at_rates = bounds[worst]
    limit = misclassification_limit(pst, worst)
    if bound <= limit:
        return ADMITTED, None
    if at_rates > limit:
        return DRIFTED, worst
    return UNCERTIFIED, worst


def alignment_size(pst, populations, limit) -> int:
    """Fewest prefixes at which a cut that misclassifies nothing and halves a
    population, read at the round's rates boundary -/+ signal, gets a
    misclassified_bounds bound of at most limit at a look's level: the size a
    population is first drawn at.  Sizing only."""
    signal = pst.config.min_signal_strength
    rates = (pst.decision_boundary + signal, pst.decision_boundary - signal)
    size = 2
    while True:
        half = size // 2
        counts = ((round(rates[0] * half), half), (round(rates[1] * half), half))
        bound, _ = misclassified_bounds(
            pst, {UNIFORM: counts}, ACCEPT_PRESERVING_ERROR_RATE / (2 * populations)
        )[UNIFORM]
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


def prefixes_to_certify(pst, counts, label, drawn, vs) -> int:
    """How many more prefixes to draw for label alone, to settle a bound the
    drawn prefixes in hand left undecided.

    How many it takes depends on the rates, so the rates in hand are the guess:
    if the same ones held over twice the counts, or three times, would label's
    misclassified_bounds come out decided -- its bound within its
    misclassification_limit, or its share at the rates read past it?  The first multiple that would is
    the answer.  Only label is drawn from, so only its counts grow.
    """
    budget = certification_budget(pst, vs)
    empty = ((0, 0), (0, 0))
    level = ACCEPT_PRESERVING_ERROR_RATE / 2
    limit = misclassification_limit(pst, label)
    for multiple in range(2, 2 + budget // drawn):
        supposed = {
            **counts,
            label: tuple(
                (hits * multiple, n * multiple) for hits, n in counts.get(label, empty)
            ),
        }
        bound, at_rates = misclassified_bounds(pst, supposed, level)[label]
        if bound <= limit or at_rates > limit:
            return drawn * (multiple - 1)
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
            prefixes_to_certify(pst, counts, label, max(1, len(held)), voters),
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
        # Two looks, each at half the rate.
        level = ACCEPT_PRESERVING_ERROR_RATE / 2
        verdict, blamed = drift_verdict(pst, counts, level)
        if verdict is UNCERTIFIED:
            counts = self._certify_further(pst, counts, blamed, voters)
            verdict, blamed = drift_verdict(pst, counts, level)
        if verdict is ADMITTED:
            return ADMITTED, None
        if verdict is DRIFTED:
            # A veto scores the population's own reads, which every later family
            # would read again: kept, one unlucky sample refuses each sound family
            # in turn until the search gives up.
            redrawn = prefixes_for_split(pst, self._state, blamed, self._sizes[blamed])
            if redrawn:
                self._drawn[blamed] = redrawn
            else:
                self._drawn.pop(blamed, None)
        self.refusals += 1
        if self.refusals >= ACCEPT_PRESERVING_GIVE_UP:
            bound, at_rates = misclassified_bounds(pst, counts, level)[blamed]
            raise NoAcceptPreservingFamily(
                f"{self.refusals} families refused: the last misclassifies "
                f"{at_rates:.0%} of {blamed} at the rates read, at most {bound:.0%}, "
                f"against {misclassification_limit(pst, blamed):.0%}; no suffix family "
                f"realises the accept-preserving split on this target"
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
        read_rates(pst.config, pst.decision_boundary),
    )
    vs = vs[:size]
    decision = pst.compute_decision(vs, pst.table.representative)
    fnr, worst = pst.fnr_from_decision(decision)
    too_high = f"FNR {fnr:.4f} too high"
    if fnr > pst.config.fnr_limit:
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
        read_rates(pst.config, decision_boundary),
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
                read_rates(pst.config, decision_boundary),
            )
            if len(vs) >= family_size:
                break

        judged = judge_family(pst, gate, v, vs, family_size)
        if judged.fnr > pst.config.fnr_limit:
            aligned = aligned_family(pst, v, family_size)
            if len(aligned) >= family_size:
                offered = judge_family(pst, gate, v, aligned, family_size)
                if offered.fnr <= pst.config.fnr_limit:
                    judged = offered

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
