from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import tqdm.auto as tqdm

from .mask_table import MaskTable
from .sampler import Sampler
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
    min_suffix_frequency: float = 0.05
    min_acc_rej: float = 0.1


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

    def _sample_suffix(self) -> int:
        while True:
            v = self.sampler.sample(rng=self.rng, alphabet_size=self.alphabet_size)
            if self.table.contains_suffix(v):
                continue
            row = self.table.intern_suffix(v)
            # A sampled suffix is an acceptance-family candidate, so observe it
            # over the whole prefix pool: that both fills its column for
            # clustering and marks it (via "fully observed") as a family suffix,
            # as opposed to the partially-observed transition distinguishers.
            self.table.column(row)
            return row

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

    def sample_more_suffixes(self, *, amount: int):
        for _ in tqdm.trange(amount, desc="Completing suffix family", delay=1):
            self._sample_suffix()

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
