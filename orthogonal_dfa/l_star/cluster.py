from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import scipy.stats

from .mask_table import UNIFORM
from .prefix_populations import grow_population, population_labels, prefixes_for_split
from .statistics import (
    evidence_margin_for_population_size,
    fpr_for_coverage_error,
    low_tail_detection_size,
    population_size_and_evidence_margin,
)


def identify_cluster_around(
    pst, seed: int, count: int, decision_boundary: float
) -> Tuple[List[int], float]:
    # Cluster only over fully-observed suffix columns -- the sampled acceptance-
    # family suffixes -- to avoid forcing a bunch of additional computation on the
    # partially-observed transition distinguishers.
    #
    # Restrict to representative prefix columns: the suffix family and the
    # decision boundary are global calibration, and a caller that has re-scoped
    # them means that scope to be what calibration reads.
    candidate = pst.table.fully_observed()
    masks = pst.table.observed_masks(candidate, pst.table.representative)
    seed_local = int(np.searchsorted(candidate, seed))
    assert candidate[seed_local] == seed, "cluster seed must be fully observed"
    # Weigh each population equally in clustering
    weights = np.zeros(masks.shape[1])
    for population in pst.table.population_masks().values():
        weights[population] += 1 / population.sum()
    # Only keep clustering while the seed belongs to the cluster.
    # We want to avoid drifting the cluster center away from the seed, which can
    # happen if the seed has a very small cluster relative to `count`.
    cluster = [seed_local]
    loss = float("inf")
    while True:
        cluster_center = masks[cluster].mean(0) > decision_boundary
        losses = ((masks != cluster_center) * weights).sum(1)
        # Ties here are common, and breaking them differently each pass churns
        # the family; every suffix that joins it costs a column of queries.
        nearest = losses.argsort(kind="stable")[:count]
        if losses[seed_local] > losses[nearest[-1]]:
            break
        if seed_local not in nearest:
            # The check above did not fire, so the seed is out on a tie.
            nearest[-1] = seed_local
        new_loss = losses[nearest].sum()
        if new_loss >= loss:
            break
        cluster, loss = nearest, new_loss

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


def read_rates(config, decision_boundary):
    """The rates a family is read at.

    A caller may ask for any crispness it likes; how far the split then sits from
    accept preservation is bounded by the band those rates buy, so the rate asked
    for is used only where it already holds ``max_coverage_error``.
    """
    return (
        min(
            config.acceptable_fpr,
            fpr_for_coverage_error(
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
    acceptable_fpr, acceptable_fnr = rates
    size, _ = population_size_and_evidence_margin(
        min_signal_strength, acceptable_fpr, acceptable_fnr, center=decision_boundary
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
    acceptable_fpr, acceptable_fnr = rates
    for size in range(have, smallest - 1, -1):
        found = evidence_margin_for_population_size(
            min_signal_strength,
            acceptable_fpr,
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

#: Chance of calling a family drifted when it is not, or clean when it is not.
ACCEPT_PRESERVING_ERROR_RATE = 0.05

#: What the gate was able to conclude about a family.
ADMITTED, DRIFTED, UNCERTIFIED = "admitted", "drifted", "uncertified"


class NoAcceptPreservingFamily(Exception):
    """No accept-preserving suffix family could be sampled for this target."""


def certification_sample(pst, vs, by_population):
    """``label -> (family means, split column)`` for prefixes read only to
    settle the split, and never added to the table.

    Reading one costs a query per family member, plus the one for the split
    itself.  Adding it to the table instead costs a query per fully observed
    column -- an order of magnitude more once the pool has grown -- and it
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
    """``label -> ((hits, n), (hits, n))``, the accept and reject sides of the
    cut counted on the split's own column, one entry per prefix population.

    A population holds one class or both, so a side of ``n = 0`` is ordinary.
    """
    return {
        label: tuple(
            (int(column[side].sum()), int(side.sum()))
            for side in (decision >= pst.accept_thresh, decision < pst.reject_thresh)
        )
        for label, (decision, column) in reads.items()
    }


def _sides(counts):
    """The sides of ``counts`` that hold prefixes, as ``(kind, hits, n)``.

    A side of ``n = 0`` is not a side read and failed: at that size neither test
    can clear its level, so counting it refuses every admit.
    """
    return [
        (kind, hits, n) for kind, (hits, n) in zip(("accept", "reject"), counts) if n
    ]


def drift_verdict(pst, by_population):
    """``(verdict, label)``: whether the family cuts with the classes, against
    them, or not readably -- and which population says so.

    Membership of ``p + v`` is membership of ``p`` for the empty suffix, so the
    split's column says what the oracle makes of the prefixes themselves.

    Any population may veto, only the uniform one may admit.  A population read
    as the class it is not says the family drifted, whatever else reads right --
    a state's prefixes are one class, so a backwards reading puts every one of
    them where the oracle contradicts it.  Separating the classes *at all* is a
    claim about the distribution the learner is scored on, and the pool is the
    only population drawn from it.

    Drift is read first: a family can separate the classes on the pool and still
    invert a state.  The label is what the search grows to answer the refusal,
    so an unreadable split names the pool, which is the one that could admit it.
    """
    alpha = ACCEPT_PRESERVING_ERROR_RATE
    sides = {label: _sides(counts) for label, counts in by_population.items()}
    num_tests = sum(len(held) for held in sides.values())
    if not num_tests:
        return UNCERTIFIED, UNIFORM

    def rejects_null(kind, hits, n, level):
        if kind == "accept":
            return scipy.stats.binom.sf(hits - 1, n, pst.accept_thresh) <= level
        return scipy.stats.binom.cdf(hits, n, pst.reject_thresh) <= level

    def drifted(kind, hits, n, level):
        if kind == "accept":
            return scipy.stats.binom.cdf(hits, n, pst.accept_thresh) <= level
        return scipy.stats.binom.sf(hits - 1, n, pst.reject_thresh) <= level

    # Shared out between the sides, so saying drifted at all costs half the rate
    # however many are read.
    for label, held in sides.items():
        if any(drifted(*side, alpha / num_tests) for side in held):
            return DRIFTED, label
    pool = sides.get(UNIFORM, [])
    if pool and all(rejects_null(*side, alpha) for side in pool):
        return ADMITTED, None
    return UNCERTIFIED, UNIFORM


def veto_size(pst, populations) -> int:
    """Prefixes a population needs for a veto to catch a family read backwards.

    An inverted accept side holds reject-class prefixes, which the oracle reads
    at ``reject_thresh`` rather than at nothing, so the size is asked for the
    miss rate and not the level alone.
    """
    # Two sides apiece, which is the most `drift_verdict` can share the rate
    # between: sizing for more of them than it reads only oversizes the draw.
    level = ACCEPT_PRESERVING_ERROR_RATE / (2 * populations)
    return max(
        low_tail_detection_size(null, alt, level, ACCEPT_PRESERVING_ERROR_RATE)
        # The reject side rejects high, which is the same test on ``n - hits``.
        for null, alt in (
            (pst.accept_thresh, pst.reject_thresh),
            (1 - pst.reject_thresh, 1 - pst.accept_thresh),
        )
    )


def certification_budget(pst, vs) -> int:
    """Never more prefixes than the round of pooled prefixes this stands in for
    would have cost.  One of those spends a query on every fully observed
    column, where one read for the split spends a query per family member and
    one for the split itself, so the budget in prefixes is the ratio between
    them.
    """
    columns = max(1, len(pst.table.fully_observed()))
    return max(1, pst.config.num_addtl_prefixes * columns // (len(vs) + 1))


def prefixes_to_certify(pst, counts, drawn, vs) -> int:
    """How many more prefixes to draw for the split alone, to settle a verdict
    the ``drawn`` prefixes in hand left undecided.

    How many it takes depends on the rates, so the rates in hand are the guess:
    if the same ones held over twice the counts, or three times, would the
    verdict come out decided?  The first multiple that would is the answer.

    Only the uniform pool is drawn from, so only its counts grow with the
    multiple.  Scaling the rest would be asking what a draw nobody makes would
    say.
    """
    budget = certification_budget(pst, vs)
    empty = ((0, 0), (0, 0))
    for multiple in range(2, 2 + budget // drawn):
        supposed = {
            **counts,
            UNIFORM: tuple(
                (hits * multiple, n * multiple)
                for hits, n in counts.get(UNIFORM, empty)
            ),
        }
        if drift_verdict(pst, supposed)[0] is not UNCERTIFIED:
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

    def _certification_prefixes(self, pst, voters):
        """``label -> prefixes`` to certify a family over, drawn once for the
        round and read by every family it tries."""
        if self._drawn is None:
            labels = population_labels(self._state)
            pool = min(
                max(1, int(pst.table.representative.sum())),
                certification_budget(pst, voters),
            )
            veto = veto_size(pst, len(labels))
            drawn = {
                label: prefixes_for_split(
                    pst, self._state, label, pool if label == UNIFORM else veto
                )
                for label in labels
            }
            self._drawn = {label: held for label, held in drawn.items() if held}
        return self._drawn

    def _certify_further(self, pst, counts, voters):
        """``counts`` with a further read of the uniform pool added into it."""
        held = self._drawn[UNIFORM]
        more = prefixes_for_split(
            pst,
            self._state,
            UNIFORM,
            prefixes_to_certify(pst, counts, len(held), voters),
        )
        if not more:
            return counts
        # Kept, so a later family is read on these rather than buying them again.
        self._drawn[UNIFORM] = held + more
        extra = _split_counts(pst, certification_sample(pst, voters, {UNIFORM: more}))
        empty = ((0, 0), (0, 0))
        return {
            **counts,
            UNIFORM: tuple(
                (hits + grown_hits, n + grown_n)
                for (hits, n), (grown_hits, grown_n) in zip(
                    counts.get(UNIFORM, empty), extra.get(UNIFORM, empty)
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
        verdict, blamed = drift_verdict(pst, counts)
        if verdict is UNCERTIFIED:
            counts = self._certify_further(pst, counts, voters)
            verdict, blamed = drift_verdict(pst, counts)
        if verdict is ADMITTED:
            return ADMITTED, None
        self.refusals += 1
        if self.refusals >= ACCEPT_PRESERVING_GIVE_UP:
            hits_a, n_a = counts[blamed][0]
            hits_r, n_r = counts[blamed][1]
            read = "read" if verdict is DRIFTED else "could not be read"
            raise NoAcceptPreservingFamily(
                f"{self.refusals} families running {read} as cutting against the "
                f"classes: the last put {hits_a / max(n_a, 1):.0%} of the prefixes it "
                f"accepts and {hits_r / max(n_r, 1):.0%} of those it rejects on the "
                f"accepting side of the empty suffix, against thresholds of "
                f"{pst.accept_thresh:.0%} and {pst.reject_thresh:.0%}; no suffix "
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
        read_rates(pst.config, pst.decision_boundary),
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
                pst.config.min_signal_strength,
                decision_boundary,
                read_rates(pst.config, decision_boundary),
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
