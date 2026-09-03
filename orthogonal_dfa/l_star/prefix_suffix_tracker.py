from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import tqdm.auto as tqdm

from .mask_table import MaskTable
from .sampler import Sampler
from .statistics import binomial_side_of_boundary
from .structures import Oracle

#: Below this a signal is not worth sizing a population for.
MIN_SIGNAL_STRENGTH = 0.001


@dataclass
class SearchConfig:
    suffix_size_counterexample_gen: int
    min_signal_strength: float
    num_addtl_prefixes: Optional[int] = None
    fnr_limit: float = 0.02
    split_pval: float = 0.001
    min_suffix_frequency: float = 0.02
    #: Chance of screening out a suffix that does belong, spent across the
    #: whole staircase rather than per test.
    screening_alpha: float = 0.1
    #: Require the suffix family to be accept-preserving.  Only meaningful where
    #: such a family exists, which is the class-preserving precondition; a caller
    #: learning a target that fails it turns this off.
    require_accept_preserving: bool = True

    def __post_init__(self):
        # Population size goes as 1/signal^2, so a signal much below this asks for
        # one no suffix family could hold, and the search doubles N looking for it.
        assert self.min_signal_strength > MIN_SIGNAL_STRENGTH, self.min_signal_strength


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
        prefixes = [
            sampler.sample(rng, alphabet_size=oracle.alphabet_size)
            for _ in range(num_prefixes)
        ]
        return cls(
            sampler=sampler,
            rng=rng,
            oracle=oracle,
            config=config,
            table=MaskTable(oracle, prefixes),
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
        """The rows still explicable as ``reference`` plus per-cell noise, which
        flips one of the two observations at rate ``2*eta*(1-eta)``."""
        eta = 0.5 - self.config.min_signal_strength
        same_family_rate = 2 * eta * (1 - eta)
        ref = self.table.column(reference)
        candidates = np.flatnonzero(self.table.representative)
        order = candidates[self.rng.permutation(len(candidates))]
        staircase = self._screening_staircase(len(order))
        alpha = self.config.screening_alpha / len(staircase)
        alive = list(rows)
        for p in staircase:
            if not alive:
                break
            subset = np.zeros(self.num_prefixes, dtype=bool)
            subset[order[:p]] = True
            disagreements = (
                self.table.observed_masks(alive, subset) != ref[subset]
            ).sum(1)
            alive = [
                row
                for row, count in zip(alive, disagreements)
                if not binomial_side_of_boundary(
                    int(count), p, same_family_rate, failure_prob=alpha
                )
            ]
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
        )

    def fnr_from_decision(self, decision) -> float:
        """``compute_fnr`` for a decision vector already in hand."""
        arr = np.array(
            [decision < self.reject_thresh, decision >= self.accept_thresh]
        ).mean(1)
        if arr.min() == 0:
            return 1
        return 1 - arr.sum()

    def sample_more_prefixes(self):
        # Sample random prefixes and add them
        new_prefixes = set()
        while len(new_prefixes) < self.config.num_addtl_prefixes:
            prefix = self.sampler.sample(self.rng, alphabet_size=self.alphabet_size)
            if prefix in new_prefixes or self.table.contains_prefix(prefix):
                continue
            new_prefixes.add(prefix)
        self.table.add_prefixes(sorted(new_prefixes))

    def sample_more_suffixes(self, *, amount: int, reference: Optional[int] = None):
        """Grow the pool of clustering candidates by ``amount`` suffixes that
        survive screening against ``reference``."""
        kept = 0
        drawn = 0
        max_draws = int(np.ceil(amount / self.config.min_suffix_frequency))
        every = np.ones(self.num_prefixes, dtype=bool)
        with tqdm.tqdm(total=amount, desc="Completing suffix family", delay=1) as pbar:
            while kept < amount and drawn < max_draws:
                cohort = self._draw_cohort(min(amount, max_draws - drawn))
                drawn += len(cohort)
                survivors = (
                    cohort
                    if reference is None
                    else self._screen_cohort(cohort, reference)
                )
                if survivors:
                    # The dropped ones stay partial, keeping them out of
                    # fully_observed() and so out of add_prefixes' top-ups.
                    self.table.observed_masks(survivors, every)
                kept += len(survivors)
                pbar.update(len(survivors))
        return kept

    def compute_decision(self, vs, subset_prefixes) -> np.ndarray:
        """Mean over the suffix rows ``vs`` of the membership matrix, restricted
        to ``subset_prefixes``; the table fills any cells not yet observed."""
        return self.table.observed_masks(vs, subset_prefixes).mean(0)
