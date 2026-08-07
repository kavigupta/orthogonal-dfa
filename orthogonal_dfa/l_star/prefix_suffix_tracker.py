from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import tqdm.auto as tqdm

from .mask_table import MaskTable
from .sampler import Sampler
from .statistics import binomial_side_of_boundary
from .structures import Oracle


def short_prefix_closure(
    prefixes: List[List[int]], max_length: int, max_count: int
) -> List[List[int]]:
    """The ``max_count`` shortest distinct prefixes (including the empty string)
    of length at most ``max_length`` of any string in ``prefixes``.

    State discovery represents each state by the prefixes that *end* in it, and
    denoises by aggregating over that population.  Random length-L probe strings
    almost never end in a *transient* state (one only reachable near the start of
    a string) — e.g. the initial state, reachable only by the empty string — so
    such states get zero rows and are never discovered, capping synthesis below
    the true state count.  Seeding the prefix set with this short prefix-closed
    core gives those transient states access strings, using only short prefixes
    (the membership queries themselves remain ``prefix + suffix``, i.e. full
    length); the recurrent states are already covered by the probe strings.

    The shortest ``max_count`` are kept: every core prefix is queried against
    every suffix, so a large core multiplies synthesis cost, while transient
    states are shallow (reachable in a few steps), so the short prefixes are both
    the cheap and the useful ones.  Keeping the shortest prefixes preserves the
    prefix-closure property (all shorter prefixes are retained).
    """
    closure = set()
    for prefix in prefixes:
        for k in range(min(len(prefix), max_length) + 1):
            closure.add(tuple(prefix[:k]))
    # Sort for a deterministic order: set-iteration order of tuples varies with
    # the CPython version (tuple hashing changed across 3.x), which would make
    # the prefix list — and the noisy statistics computed over it — depend on
    # the interpreter.  Order by (length, contents) so the empty string is first.
    ordered = sorted(closure, key=lambda p: (len(p), p))
    return [list(p) for p in ordered[:max_count]]


@dataclass
class SearchConfig:
    suffix_family_size: int
    evidence_margin: float
    decision_rule_fpr: float
    suffix_size_counterexample_gen: int
    min_signal_strength: float
    num_addtl_prefixes: Optional[int] = None
    fnr_limit: float = 0.02
    split_pval: float = 0.001
    min_suffix_frequency: float = 0.02
    #: Chance of screening out a suffix that does belong to the family.  Spent
    #: across the whole staircase, not per test.  Raising it rejects more of the
    #: population on fewer prefixes; the cost of a mistake is one extra draw.
    screening_alpha: float = 0.1


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

    def __post_init__(self):
        if self.evidence_margin == 0.0:
            self.evidence_margin = self.config.evidence_margin

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
        prefix_core_length: int = 4,
        prefix_core_size: int = 32,
    ) -> "PrefixSuffixTracker":
        prefixes = [
            sampler.sample(rng, alphabet_size=oracle.alphabet_size)
            for _ in range(num_prefixes)
        ]
        # Per-prefix flag: True for "representative" probe prefixes (drawn from
        # the sampler), False for the short prefix-closed core.  Global
        # calibration (decision boundary, FNR) is computed over representative
        # prefixes only, so the statistically-unrepresentative core does not bias
        # it; state discovery still uses every prefix so transient states split.
        representative = [True] * len(prefixes)
        if prefix_core_length > 0 and prefix_core_size > 0:
            existing = {tuple(p) for p in prefixes}
            core = [
                p
                for p in short_prefix_closure(
                    prefixes, prefix_core_length, prefix_core_size
                )
                if tuple(p) not in existing
            ]
            prefixes = prefixes + core
            representative = representative + [False] * len(core)
        return cls(
            sampler=sampler,
            rng=rng,
            oracle=oracle,
            config=config,
            table=MaskTable(oracle, prefixes, representative),
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
        """The rows whose membership pattern is still explicable as
        ``reference``'s plus per-cell noise, after testing the whole cohort on a
        growing subset of prefixes and dropping each row at the first subset that
        settles it.

        A suffix in the same family as ``reference`` induces the same accept /
        reject labelling of the prefixes, so the two columns differ only where
        one of the two queries was flipped -- a rate of ``2*eta*(1-eta)``, with
        ``eta`` the per-query noise.  ``min_signal_strength`` lower-bounds the
        signal, hence upper-bounds ``eta``, so this null holds without knowing
        anything about the family being assembled; a candidate significantly
        above it is not in the family and does not need its remaining cells.

        The cohort shares one prefix ordering, which is what lets a whole
        staircase step go to the oracle as a single batch -- one call per step
        rather than one per candidate.  Each row is still judged only on its own
        disagreement count, so the decisions are the ones a per-candidate screen
        would make.

        Only representative prefixes are used, matching the population the
        clustering and the decision boundary are calibrated over.
        """
        eta = 0.5 - self.config.min_signal_strength
        same_family_rate = 2 * eta * (1 - eta)
        ref = self.table.column(reference)
        candidates = np.flatnonzero(self.table.representative)
        order = candidates[self.rng.permutation(len(candidates))]
        staircase = self._screening_staircase(len(order))
        # The staircase gets one shot at rejecting per step, so the per-test rate
        # has to be the budget split across them for screening_alpha to mean what
        # it says.
        alpha = self.config.screening_alpha / len(staircase)
        alive = list(rows)
        for p in staircase:
            if not alive:
                break
            subset = np.zeros(self.num_prefixes, dtype=bool)
            subset[order[:p]] = True
            disagreements = (self.table.observed_masks(alive, subset) != ref[subset]).sum(1)
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

        Computed over the representative prefixes only: the short prefix-closed
        core exists to give transient states discovery rows, not to recalibrate
        the family against an unrepresentative population.
        """
        decision = self.compute_decision(vs, self.table.representative)
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
            prefix = tuple(
                self.sampler.sample(self.rng, alphabet_size=self.alphabet_size)
            )
            if prefix in new_prefixes or self.table.contains_prefix(list(prefix)):
                continue
            new_prefixes.add(prefix)
        self.table.add_prefixes(sorted(list(x) for x in new_prefixes))

    def sample_more_suffixes(self, *, amount: int, reference: Optional[int] = None):
        """Grow the pool of clustering candidates by ``amount``.

        ``amount`` counts survivors, not draws: the caller loops until the pool
        supports a family of ``suffix_family_size`` and treats a round that
        grows it less as failure to progress, so a screened round has to deliver
        as much as an unscreened one did -- it just pays a screening subset per
        rejection instead of a full column.

        Draws are capped using ``min_suffix_frequency``, the assumed floor on
        how often a family member comes up, so a reference that nothing matches
        cannot spin here forever.

        Candidates are drawn and screened a cohort at a time so each staircase
        step is one oracle call for the whole cohort.  A cohort's survivors are
        all completed even if that overshoots ``amount``: they passed, and
        stranding a passed suffix half-observed would waste the screening
        already paid for and lose it for good, since it can never be redrawn.
        """
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
                    # Fully observe them, so they can be used in clustering.  The
                    # ones the screen dropped keep the cells it paid for and stay
                    # out of fully_observed(), so they are neither clustering
                    # candidates nor columns add_prefixes has to top up.
                    self.table.observed_masks(survivors, every)
                kept += len(survivors)
                pbar.update(len(survivors))
        return kept

    def compute_decision(self, vs, subset_prefixes) -> np.ndarray:
        """Mean over the suffix rows ``vs`` of the membership matrix, restricted
        to ``subset_prefixes``; the table fills any cells not yet observed."""
        return self.table.observed_masks(vs, subset_prefixes).mean(0)

    def compute_decision_from_strings(
        self, vs: List[List[int]], subset_prefixes=None
    ) -> np.ndarray:
        if subset_prefixes is None:
            subset_prefixes = np.ones(self.num_prefixes, dtype=bool)
        vs_idxs = [self.table.intern_suffix(v) for v in vs]
        return self.compute_decision(vs_idxs, subset_prefixes)
