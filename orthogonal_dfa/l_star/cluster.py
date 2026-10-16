import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

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
    """The first count of the rest of the pool in nearest_to_anchor_group's
    order, anchored on seed's reads.  The seed anchors but does not vote: with
    the empty seed voting, reading z at a midfix m would read the very string
    the read of z·m reads at the root, so node reads would share noise."""
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
    cluster = []
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
        cluster += others[order[:count]].tolist()
    if not cluster:
        return [], decision_boundary

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

#: P(admitting, at some family of a search, one whose cut misclassifies more than
#: misclassification_limit of some population).
ACCEPT_PRESERVING_ERROR_RATE = 0.05
#: The tails a family spends per population: three counts on each side of the
#: uniform pool's calibration sample, three on each side of a look at the
#: population, and the passage from the draws to the population.
GATE_TAILS = 13

#: What the gate was able to conclude about a family.
ADMITTED, DRIFTED, UNCERTIFIED = "admitted", "drifted", "uncertified"


class NoAcceptPreservingFamily(Exception):
    """No accept-preserving suffix family could be sampled for this target."""


def hoeffding_deviation(n, level) -> float:
    """t with P(a sum of n independent reads in [0, 1] strays past its mean by t) <= level."""
    return math.sqrt(n * math.log(1 / level) / 2)


def size_level(level, n) -> float:
    """level / ((n + 1)(n + 2)), which sums to under level over every n."""
    return level / ((n + 1) * (n + 2))


@dataclass(frozen=True)
class Side:
    """The prefixes on one side of a family's cut: how many, how many the seed
    reads against the side, and the fewest any reference disagrees with the seed
    on."""

    size: int
    against: int
    apart: int

    def scaled(self, m):
        return Side(m * self.size, m * self.against, m * self.apart)


@dataclass(frozen=True)
class Sample:
    """A population's certification prefixes, read against a family: the side
    its cut accepts, the side it rejects, and how many draws finding them took."""

    accept: Side
    reject: Side
    drawn: int

    @property
    def size(self) -> int:
        return self.accept.size + self.reject.size

    def scaled(self, m):
        return Sample(self.accept.scaled(m), self.reject.scaled(m), m * self.drawn)


def certification_sample(pst, vs, refs, by_population):
    """label -> (family means, split column, reference reads) for prefixes
    read only to settle the split, and never added to the table.

    Reading one costs a query per family member and per reference, plus the one
    for the split itself.  Adding it to the table instead costs as much and
    unsettles the FNR the round has only just met, which is bought back with a
    fresh cohort of suffixes that every later prefix is then read against.
    """
    family = [pst.table.suffix(v) for v in vs]
    references = [pst.table.suffix(v) for v in refs]
    out = {}
    for label, prefixes in by_population.items():
        pairs = [p + sfx for p in prefixes for sfx in family]
        ref_pairs = [p + sfx for sfx in references for p in prefixes]
        read = pst.table.memo.membership_queries(pairs + ref_pairs + prefixes)
        n_family, n_refs = len(pairs), len(ref_pairs)
        out[label] = (
            np.asarray(read[:n_family]).reshape(len(prefixes), len(family)).mean(1),
            np.asarray(read[n_family + n_refs :]),
            np.asarray(read[n_family : n_family + n_refs]).reshape(
                len(references), len(prefixes)
            ),
        )
    return out


def read_sample(pst, reads, drawn) -> Sample:
    """The family's cut at the decision boundary, and each side's counts."""
    decision, column, references = reads
    accepted = decision >= pst.decision_boundary

    def side(on, against):
        apart = (references[:, on] != column[on]).sum(1)
        return Side(
            int(on.sum()),
            int((column[on] == against).sum()),
            int(apart.min()) if len(apart) else 0,
        )

    return Sample(side(accepted, 0), side(~accepted, 1), drawn)


def wrong_cap(side, deviation, rate) -> float:
    """An upper bound on the share of side the cut puts there wrongly, the
    wrong class's noise rate being at most rate and the side's two counts
    within deviation of their means.  With t_p the chance the seed reads p
    against the side, r_p p's noise rate and c the side's against share,

        (1/n) sum_p (t_p - c)^2 = c (1 - c) + (1 - 2c)(mean t - c) - (1/n) sum_p r_p (1 - r_p),

    a reference that preserves the classes disagrees with the seed at
    2 r_p (1 - r_p), and a wrong prefix contributes at least (1 - rate - c)^2."""
    n = side.size
    if n == 0:
        return 1.0
    c = side.against / n
    gap = 1 - rate - c
    if gap <= 0:
        return 1.0
    centre = max(
        (1 - 2 * c) * ((side.against - deviation) / n - c),
        (1 - 2 * c) * ((side.against + deviation) / n - c),
    )
    spread = c * (1 - c) + centre - (side.apart - deviation) / (2 * n)
    return min(1.0, max(0.0, spread) / gap**2)


def rate_bound(side, deviation, eta, cap) -> float:
    """An upper bound on the noise rate of the class side holds rightly, from
    its against share and a cap on its wrong share; eta bounds both rates."""
    if cap >= 1:
        return eta
    return min(eta, (side.against + deviation) / side.size / (1 - cap))


def calibrate(pst, uniform, level):
    """Upper bounds on the noise rates in and out of the language, read off the
    uniform pool's sample: the side its cut accepts holds the language rightly,
    the side it rejects the complement."""
    eta = 1 / 2 - pst.config.min_signal_strength

    def rate(side):
        deviation = hoeffding_deviation(side.size, level)
        return rate_bound(side, deviation, eta, wrong_cap(side, deviation, eta))

    return rate(uniform.accept), rate(uniform.reject)


def miscut_bound(sample, level, rates) -> float:
    """An upper bound on how many of the sample the cut puts on the wrong side:
    the accept side's wrong prefixes lie outside the language, at the rate out,
    and the reject side's inside it, at the rate in."""
    rate_in, rate_out = rates
    return sum(
        side.size * wrong_cap(side, hoeffding_deviation(side.size, level), rate)
        for side, rate in ((sample.accept, rate_out), (sample.reject, rate_in))
    )


def misclassified_bound(sample, level, rates) -> float:
    """What the gate holds against the limit: the miscut bound at the sample's
    size, the draws that did not turn up a fresh prefix, and the passage from
    the draws to the population, as a share of the draws."""
    n, drawn = sample.size, sample.drawn
    if drawn == 0:
        return math.inf
    return (
        miscut_bound(sample, size_level(level, n), rates)
        + (drawn - n)
        + hoeffding_deviation(drawn, size_level(level, drawn))
    ) / drawn


def misclassified_point(sample, rates) -> float:
    """The share of the sample's prefixes the cut misplaces, taking each side's
    against share and the rates read as exact: a side with a share e wrong
    reads against itself at its right class's rate plus e times the gap."""
    rate_in, rate_out = rates
    gap = 1 - rate_in - rate_out
    if sample.size == 0 or gap <= 0:
        return 0.0
    wrong = sum(
        side.size * min(1.0, max(0.0, (side.against / side.size - right) / gap))
        for side, right in ((sample.accept, rate_in), (sample.reject, rate_out))
        if side.size
    )
    return wrong / sample.size


def misclassification_limit(pst, label) -> float:
    return pst.config.max_coverage_error if label == UNIFORM else 1 / 2


def _population_verdict(pst, label, sample, level, rates):
    """ADMITTED if the bound is within limit, else DRIFTED if the point is past
    it, else UNCERTIFIED, for limit = misclassification_limit."""
    limit = misclassification_limit(pst, label)
    if misclassified_bound(sample, level, rates) <= limit:
        return ADMITTED
    if misclassified_point(sample, rates) > limit:
        return DRIFTED
    return UNCERTIFIED


def uniform_rates(pst, calibration, level):
    """The noise rates read off calibration, the uniform pool's sample, at
    its size; None where it holds no prefix."""
    if calibration is None or not calibration.size:
        return None
    return calibrate(pst, calibration, size_level(level, calibration.size))


def drift_verdict(pst, samples, rates, level):
    """(_population_verdict of j, j) for the j whose bound is furthest past its
    limit; None for j when ADMITTED."""
    if rates is None or not samples:
        return UNCERTIFIED, UNIFORM
    worst = max(
        samples,
        key=lambda label: misclassified_bound(samples[label], level, rates)
        - misclassification_limit(pst, label),
    )
    verdict = _population_verdict(pst, worst, samples[worst], level, rates)
    return verdict, None if verdict is ADMITTED else worst


def family_level(populations, attempt) -> float:
    """What a search's attempt-th family spends of ACCEPT_PRESERVING_ERROR_RATE
    on each tail at each population."""
    return size_level(
        ACCEPT_PRESERVING_ERROR_RATE / (GATE_TAILS * populations), attempt
    )


def read_right(pst, size) -> Sample:
    """A sample of size prefixes, half each side of a cut that misplaces
    none, every count at its mean at the round's rates."""
    p_0, p_1 = round_rates(pst)
    half = size // 2

    def side(rate):
        return Side(half, round(rate * half), round(2 * rate * (1 - rate) * half))

    return Sample(side(1 - p_1), side(p_0), 2 * half)


def alignment_size(pst, populations, limit) -> int:
    """Fewest prefixes, a power of 2, on which the first family of a search
    certifies a population with limit `limit` when every sample, the uniform
    pool's among them, is read right at its means.  `populations` is how many
    populations share the error rate."""
    level = family_level(populations, 0)
    size = 2
    while True:
        sample = read_right(pst, size)
        rates = calibrate(pst, sample, size_level(level, size))
        if misclassified_bound(sample, level, rates) <= limit:
            return size
        size *= 2


def prefixes_to_certify(pst, sample, rates, *, label, level, budget) -> int:
    """n (m - 1), n label's prefixes drawn, for the smallest m >= 2 at which
    label's sample times m is not UNCERTIFIED by _population_verdict; at most
    budget."""
    if sample is None or not sample.drawn or rates is None:
        return budget
    for m in range(2, 2 + budget // sample.drawn):
        scaled = sample.scaled(m)
        if _population_verdict(pst, label, scaled, level, rates) is not UNCERTIFIED:
            return sample.drawn * (m - 1)
    return budget


class AcceptPreservingGate:
    """Holds each suffix family to the accept-preserving split, across the loop
    that resamples until one passes.  Carries the give-up budget, spent on every
    round that does not produce a family the split can be certified on, and the
    count of families judged, which the error rate is spread over.

    Nothing resets either: admitting a family is the round returning, and the
    gate is made afresh for the next search."""

    def __init__(self, config, state):
        self.enabled = config.require_accept_preserving
        self.refusals = 0
        self.attempts = 0
        self._state = state
        self._drawn = None
        self._raw = None
        self._sizes = None

    def _certification_prefixes(self, pst):
        """``label -> prefixes`` to certify a family over, drawn for the round
        and read by every family it tries."""
        if self._drawn is None:
            labels = population_labels(self._state)
            self._sizes = {
                label: alignment_size(
                    pst, len(labels), misclassification_limit(pst, label)
                )
                for label in labels
            }
            self._drawn, self._raw = {}, {}
            for label in labels:
                self._draw(pst, label)
        return self._drawn

    def _draw(self, pst, label):
        held, drawn = prefixes_for_split(
            pst, self._state, label, self._sizes[label], ()
        )
        if held:
            self._drawn[label], self._raw[label] = held, drawn
        else:
            self._drawn.pop(label, None)
            self._raw.pop(label, None)

    def _certify_further(self, pst, samples, label, *, rates, readers, level):
        """samples with a further look at label's population added into it, to
        at most twice its first; readers is the family and the references."""
        held = self._drawn.get(label, [])
        wanted = prefixes_to_certify(
            pst,
            samples.get(label),
            rates,
            label=label,
            level=level,
            budget=self._sizes[label],
        )
        more, drawn = prefixes_for_split(pst, self._state, label, wanted, held)
        if not more:
            return samples
        # Kept, so a later family is read on these rather than buying them again.
        self._drawn[label] = held + more
        self._raw[label] = self._raw.get(label, 0) + drawn
        reads = certification_sample(pst, *readers, {label: self._drawn[label]})
        return {**samples, label: read_sample(pst, reads[label], self._raw[label])}

    def verdict(self, pst, seed_row, vs):
        """``(verdict, label)``: what the split says, and which population said
        it, for the search to answer."""
        if not self.enabled:
            return ADMITTED, None
        # The family was clustered over the table's prefixes, and a large enough
        # pool fits their noise, so only prefixes it never saw can test it.  The
        # seed votes on p with the very read of p being scored, so it sits out,
        # and the references are the pool's suffixes outside the family.
        voters = [u for u in vs if u != seed_row]
        refs = [u for u in pst.suffix_pool if u not in vs and u != seed_row]
        prefixes = self._certification_prefixes(pst)
        level = family_level(len(self._sizes), self.attempts)
        self.attempts += 1
        reads = certification_sample(pst, voters, refs, prefixes)
        samples = {
            label: read_sample(pst, read, self._raw[label])
            for label, read in reads.items()
        }
        rates = uniform_rates(pst, samples.get(UNIFORM), level)
        verdict, blamed = drift_verdict(pst, samples, rates, level)
        if verdict is UNCERTIFIED and blamed in samples:
            samples = self._certify_further(
                pst, samples, blamed, rates=rates, readers=(voters, refs), level=level
            )
            verdict, blamed = drift_verdict(pst, samples, rates, level)
        if verdict is ADMITTED:
            return ADMITTED, None
        if verdict is DRIFTED:
            # A refusal scores the population's own reads, which every later family
            # would read again: kept, one unlucky sample refuses each sound family
            # in turn until the search gives up.
            self._draw(pst, blamed)
        self.refusals += 1
        if self.refusals >= ACCEPT_PRESERVING_GIVE_UP:
            if blamed in samples and rates is not None:
                sample = samples[blamed]
                read = (
                    f"misclassifies {misclassified_point(sample, rates):.0%} of "
                    f"{blamed} at the rates read, at most "
                    f"{misclassified_bound(sample, level, rates):.0%}, against "
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
