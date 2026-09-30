import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import scipy.stats

from .mask_table import UNIFORM, MaskTable
from .progress import counter
from .sampler import Sampler
from .statistics import binomial_side_of_boundary
from .structures import Oracle

#: Below this a signal is not worth sizing a population for.
MIN_SIGNAL_STRENGTH = 0.001


def _low_cluster_rates(counts, sizes, failure_prob) -> List[float]:
    """For each side s, r_s with P(Binomial(n_s m, r_s) <= c_s) = failure_prob,
    for the m rows and summed counts c_s that the maximum-likelihood fit of

        w prod_s Binomial(n_s, r_lo_s) + (1 - w) prod_s Binomial(n_s, r_up_s),
        sum_s r_lo_s < sum_s r_up_s,

    to counts[s] (n_s = sizes[s]) assigns to the lower component, reached by EM
    from the split at the median of sum_s counts[s] / n_s; all the rows if no two
    components separate."""
    counts = [np.asarray(c) for c in counts]
    total = sum(c / n for c, n in zip(counts, sizes))
    low = total <= np.median(total)
    while low.any() and not low.all():
        rates = [(c[low].mean() / n, c[~low].mean() / n) for c, n in zip(counts, sizes)]
        if not all(0 < lo < 1 and 0 < up < 1 for lo, up in rates) or sum(
            lo for lo, _ in rates
        ) >= sum(up for _, up in rates):
            break
        log_lo = np.log(low.mean()) + sum(
            scipy.stats.binom.logpmf(c, n, lo)
            for c, n, (lo, _) in zip(counts, sizes, rates)
        )
        log_up = np.log(1 - low.mean()) + sum(
            scipy.stats.binom.logpmf(c, n, up)
            for c, n, (_, up) in zip(counts, sizes, rates)
        )
        relabelled = log_lo >= log_up
        if (relabelled == low).all():
            break
        low = relabelled
    if not low.any() or low.all():
        low = np.ones(len(total), dtype=bool)
    floors = []
    for c, n in zip(counts, sizes):
        hits, trials = int(c[low].sum()), n * int(low.sum())
        floors.append(
            1.0
            if hits == trials
            else float(scipy.stats.beta.ppf(1 - failure_prob, hits + 1, trials - hits))
        )
    return floors


@dataclass
class SearchConfig:
    suffix_size_counterexample_gen: int
    min_signal_strength: float
    num_addtl_prefixes: Optional[int] = None
    #: A rate every prefix population has to meet on its own, not an average
    #: across them.
    fnr_limit: float = 0.10
    #: The first two bound the split's crispness, and say nothing about whether the
    #: split is the accept-preserving one.  `acceptable_fnr` is the chance a prefix
    #: is called indecisive, all indecision counting against it; `cross_limit`
    #: bounds the chance a prefix whose rate lies outside the band is decisively
    #: called the other side.  A prefix inside the band has no such bound.
    #:
    #: `max_coverage_error` bounds instead how far the split may deviate from the
    #: true accept-preserving distinction: the share of the prefixes it decides that
    #: it decides against the denoised oracle.  The accept-preserving test holds
    #: that at `(1 - eps/signal)/2`, so asking for less asks for a wider band,
    #: bought with a tighter `cross_limit` and paid for in indecision.  Keep
    #: `acceptable_fnr` below `fnr_limit`, which holds the same indecision rate over
    #: the pool, or a clean family fails its round.
    cross_limit: float = 1.5e-7
    acceptable_fnr: float = 0.01
    max_coverage_error: float = 1 / 3
    split_pval: float = 0.001
    min_suffix_frequency: float = 0.02
    #: Chance of screening out a suffix that does belong, over every test and
    #: floor of its screening.
    screening_alpha: float = 0.1
    #: Require the suffix family to be accept-preserving.  Only meaningful where
    #: such a family exists, which is the class-preserving precondition; a caller
    #: learning a target that fails it turns this off.
    require_accept_preserving: bool = True

    def __post_init__(self):
        # Population size goes as 1/signal^2, so a signal much below this asks for
        # one no suffix family could hold, and the search doubles N looking for it.
        assert self.min_signal_strength > MIN_SIGNAL_STRENGTH, self.min_signal_strength


def _draw_budget(count: int) -> int:
    """Draws to allow in collecting ``count`` distinct strings.

    About ``count`` are needed where the sampler has far more strings than the
    pool wants, and about ``count * ln count`` where it has only half again as
    many.  Nearer exhaustion than that no fixed budget helps: the last string of
    a support of ``s`` costs ``s`` draws on its own.
    """
    return count * (1 + math.ceil(math.log(count + 1)))


def _distinct_prefixes(sampler, rng, *, alphabet_size, count, held):
    """Up to ``count`` prefixes, distinct from each other and from ``held``.

    Fewer when the sampler has fewer left to give.  Drawing until it has
    ``count`` never returns once it is out, and a pool that cannot grow is the
    caller's business rather than an error here.
    """
    drawn = set()
    for _ in range(_draw_budget(count)):
        if len(drawn) == count:
            break
        prefix = sampler.sample(rng, alphabet_size=alphabet_size)
        if prefix not in held:
            drawn.add(prefix)
    return sorted(drawn)


@dataclass
class PrefixSuffixTracker:
    """Owns the search calibration (decision boundary, evidence margin, family
    sampling) on top of a :class:`MaskTable`.

    The prefixes, suffixes and membership matrix live entirely in ``self.table``
    and are reached only through its interface -- nothing here (or in callers)
    touches the raw arrays.
    """

    sampler: Sampler
    rng: np.random.Generator
    oracle: Oracle
    config: SearchConfig
    table: MaskTable
    decision_boundary: float = 0.5
    evidence_margin: float = 0.0
    #: The suffix rows clustering picks families from.
    suffix_pool: List[int] = field(default_factory=list)

    @property
    def num_prefixes(self) -> int:
        return self.table.num_prefixes

    @property
    def alphabet_size(self) -> int:
        return self.oracle.alphabet_size

    @property
    def accept_thresh(self) -> float:
        return self.decision_boundary + self.evidence_margin

    @property
    def reject_thresh(self) -> float:
        return self.decision_boundary - self.evidence_margin

    @classmethod
    def create(
        cls,
        sampler,
        rng,
        oracle,
        config: "SearchConfig",
        *,
        num_prefixes: int,
    ) -> "PrefixSuffixTracker":
        # A string here is a byte per symbol, so a wider alphabet has nothing to
        # be written down in.  Said once, and before the first draw, rather than
        # left to surface as whichever byte conversion is reached first.
        assert oracle.alphabet_size <= 256, oracle.alphabet_size
        prefixes = _distinct_prefixes(
            sampler,
            rng,
            alphabet_size=oracle.alphabet_size,
            count=num_prefixes,
            held=(),
        )
        return cls(
            sampler=sampler,
            rng=rng,
            oracle=oracle,
            config=config,
            table=MaskTable(oracle, prefixes, population=UNIFORM),
        )

    def _screening_staircase(self, available: int) -> List[int]:
        """Prefix counts to test a candidate at, smallest first."""
        out = []
        p = 16
        while p < available:
            out.append(p)
            p *= 2
        out.append(available)
        return out

    def _screen_cohort(self, rows: List[int], reference: int) -> List[int]:
        """The rows not screened out.  At each prefix count n of the staircase,
        and on each side s in {R = 1, R = 0} of the reference over those prefixes,
        row x is screened out if its n_s disagreements D_s(x) reject

            H0: D_s(x) ~ Binomial(n_s, rho_s)

        in the upper tail at alpha, rho_s the _low_cluster_rates of every row's
        counts."""
        ref = self.table.column(reference)
        candidates = np.flatnonzero(self.table.representative)
        order = candidates[self.rng.permutation(len(candidates))]
        staircase = self._screening_staircase(len(order))
        # A test and a floor on each side at each step.
        alpha = self.config.screening_alpha / (4 * len(staircase))
        alive = list(rows)
        for p in staircase:
            if not alive:
                break
            subset = np.zeros(self.num_prefixes, dtype=bool)
            subset[order[:p]] = True
            observed = self.table.observed_masks(alive, subset)
            sides = [ref[subset] == read for read in (True, False)]
            sides = [side for side in sides if side.any()]
            counts = [(observed[:, side] != ref[subset][side]).sum(1) for side in sides]
            sizes = [int(side.sum()) for side in sides]
            rates = _low_cluster_rates(counts, sizes, alpha)
            too_far = np.zeros(len(alive), dtype=bool)
            for count, n, rate in zip(counts, sizes, rates):
                too_far |= [
                    binomial_side_of_boundary(int(c), n, rate, failure_prob=alpha)
                    is True
                    for c in count
                ]
            alive = [row for row, far in zip(alive, too_far) if not far]
        return alive

    def _draw_cohort(self, size: int) -> List[int]:
        """``size`` unseen suffixes, interned but not yet observed."""
        rows = []
        while len(rows) < size:
            v = self.sampler.sample(rng=self.rng, alphabet_size=self.alphabet_size)
            if self.table.contains_suffix(v):
                continue
            rows.append(self.table.intern_suffix(v))
        return rows

    def compute_fnr(self, vs):
        """
        Compute the false negative rate for the given suffix family vs.

        This is the % of prefixes that are neither classified as positive nor negative by the
        given suffix family.

        A special case is that if the family classifies all prefixes as positive or negative,
        then the FNR is 1 rather than 0 (since the prediction is uninformative).

        Computed over the representative prefixes only, which a caller may
        re-scope to focus the family.
        """
        return self.fnr_from_decision(
            self.compute_decision(vs, self.table.representative)
        )[0]

    def fnr_from_decision(self, decision) -> Tuple[float, Optional[object]]:
        """``compute_fnr`` for a decision vector already in hand, and which
        population it is the rate of.

        The worst population's rate, not the rate across all of them: whether a
        prefix is decisive is a property of the state it reaches, so one
        population reading high is averaged away by the rest.
        """
        decided = np.array(
            [decision < self.reject_thresh, decision >= self.accept_thresh]
        )
        if decided.mean(1).min() == 0:
            return 1, None
        indecisive = ~decided.any(0)
        rates = [
            (float(indecisive[m].mean()), label)
            for label, m in self.table.population_masks().items()
        ]
        if not rates:
            return float(indecisive.mean()), None
        # Not max(rates): a tie falls through to the labels, which are not all
        # one type.
        return max(rates, key=lambda rate_and_label: rate_and_label[0])

    def sample_more_prefixes(self):
        new_prefixes = _distinct_prefixes(
            self.sampler,
            self.rng,
            alphabet_size=self.alphabet_size,
            count=self.config.num_addtl_prefixes,
            held=set(self.table.prefixes),
        )
        if new_prefixes:
            self.table.add_prefixes(new_prefixes, population=UNIFORM)

    def sample_more_suffixes(self, *, amount: int, reference: Optional[int] = None):
        """Grow the pool of clustering candidates by ``amount`` suffixes that
        survive screening against ``reference``, returning ``(kept, drawn)``.

        A cohort is screened whole, so the last one can carry ``kept`` past
        ``amount``."""
        kept = 0
        drawn = 0
        max_draws = int(np.ceil(amount / self.config.min_suffix_frequency))
        with counter(amount, "Completing suffix family") as pbar:
            while kept < amount and drawn < max_draws:
                cohort = self._draw_cohort(min(amount, max_draws - drawn))
                drawn += len(cohort)
                survivors = (
                    cohort
                    if reference is None
                    else self._screen_cohort(cohort, reference)
                )
                self.suffix_pool.extend(survivors)
                kept += len(survivors)
                pbar.update(len(survivors))
        return kept, drawn

    def compute_decision(self, vs, subset_prefixes) -> np.ndarray:
        """Mean over the suffix rows ``vs`` of the membership matrix, restricted
        to ``subset_prefixes``; the table fills any cells not yet observed."""
        return self.table.observed_masks(vs, subset_prefixes).mean(0)
