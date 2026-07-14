from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import tqdm.auto as tqdm

from .sampler import Sampler
from .structures import Oracle


def compute_mask(for_state, oracle, v, active=None):
    """Membership of ``x + v`` for each prefix ``x`` in ``for_state``.

    When ``active`` (a boolean mask over ``for_state``) is given, only the active
    prefixes are queried; the rest are left ``NaN`` ("not evaluated").  A ``NaN`` cell
    fails both the accept and reject threshold comparisons, so it is naturally
    excluded from the split test rather than counted as either class.
    """
    if active is None:
        return np.array([oracle.membership_query(x + v) for x in for_state], np.float32)
    mask = np.full(len(for_state), np.nan, np.float32)
    for i in np.flatnonzero(active):
        mask[i] = oracle.membership_query(for_state[i] + v)
    return mask


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
    # When True, a prepended family [c]+vs is queried only over the prefixes of the
    # states its parent family split (which is where it can produce a split -- see
    # the prepend-provenance analysis), instead of over all prefixes.  Un-evaluated
    # cells are left NaN and excluded from the split test.  The transition matrix is
    # then computed by classifying only a capped subsample of each state's prefixes'
    # c-successors, rather than re-completing every [c]+predicate over all prefixes.
    restrict_prepend_queries: bool = False
    # Max prefixes per state whose c-successor is classified when estimating each
    # transition (a per-(state, symbol) majority vote); None classifies all of them.
    transition_sample_cap: Optional[int] = None
    # After a restricted-prepend discovery, iteratively re-split states whose sampled
    # c-successors straddle a real split (a cross-branch split the restriction missed),
    # un-restricting only the offending [c]+distinguisher over that state's prefixes.
    resplit_transitions: bool = False
    # Min share of a state's sampled c-successors that must land on a second distinct
    # successor before it's treated as a candidate cross-branch split.
    resplit_min_share: float = 0.15


@dataclass
class PrefixSuffixTracker:
    sampler: Sampler
    rng: np.random.Generator
    oracle: Oracle
    config: SearchConfig
    prefixes: List[List[int]]
    suffixes_seen: dict
    suffix_bank: List[List[int]]
    corresponding_masks: List[np.ndarray]
    decision_boundary: float = 0.5
    evidence_margin: float = 0.0
    # Per-prefix flag: True for "representative" probe prefixes (drawn from the
    # sampler), False for the short prefix-closed core.  Global calibration
    # (decision boundary, FNR) is computed over representative prefixes only, so
    # the statistically-unrepresentative core does not bias it; state discovery
    # still uses every prefix so transient states are split out.  None => all
    # prefixes are representative (the default with no core).
    representative_prefixes: Optional[List[bool]] = None

    def __post_init__(self):
        if self.evidence_margin == 0.0:
            self.evidence_margin = self.config.evidence_margin
        if self.representative_prefixes is None:
            self.representative_prefixes = [True] * len(self.prefixes)

    @property
    def representative(self) -> np.ndarray:
        """Boolean mask selecting the representative (non-core) prefixes."""
        return np.array(self.representative_prefixes, dtype=bool)

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
            prefixes=prefixes,
            suffixes_seen={},
            suffix_bank=[],
            corresponding_masks=[],
            representative_prefixes=representative,
        )

    def sample_suffix(self) -> Tuple[List[int], np.ndarray, int]:
        while True:
            v = self.sampler.sample(rng=self.rng, alphabet_size=self.alphabet_size)
            if tuple(v) in self.suffixes_seen:
                continue
            return self.record_suffix(v)

    def record_suffix(
        self, v: List[int], active=None
    ) -> Tuple[List[int], np.ndarray, int]:
        if tuple(v) in self.suffixes_seen:
            _, _, idx = self.suffixes_seen[tuple(v)]
            if active is not None:
                # Grow the partial column over the requested prefixes; ``active=None``
                # leaves it as-is (a partial predicate classifies its own state's
                # prefixes; the rest fall to "unclassified" and are ignored).
                live = self.corresponding_masks[idx]
                for i in np.flatnonzero(active & np.isnan(live)):
                    live[i] = self.oracle.membership_query(self.prefixes[i] + v)
            return self.suffixes_seen[tuple(v)]
        self.suffix_bank.append(v)
        mask = compute_mask(self.prefixes, self.oracle, v, active)
        self.corresponding_masks.append(mask)
        self.suffixes_seen[tuple(v)] = v, mask, len(self.suffix_bank) - 1
        return v, mask, len(self.suffix_bank) - 1

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
        arr = self.compute_decision_array_from_strings(
            [self.suffix_bank[v] for v in vs], self.representative
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
            if prefix in new_prefixes or prefix in self.prefixes:
                continue
            new_prefixes.add(prefix)
        new_prefixes = sorted(list(x) for x in new_prefixes if x not in self.prefixes)
        self.add_prefixes(new_prefixes)

    def sample_more_suffixes(self, *, amount: int):
        for _ in tqdm.trange(amount, desc="Completing suffix family", delay=1):
            self.sample_suffix()

    def corresponding_masks_for_subset(self, subset_prefixes) -> List[np.ndarray]:
        corresponding_masks = np.array(self.corresponding_masks)
        return corresponding_masks[:, subset_prefixes]

    def compute_decision(self, vs, subset_prefixes) -> np.ndarray:
        selected_masks = self.corresponding_masks_for_subset(subset_prefixes)[vs]
        return selected_masks.mean(0)

    def compute_decision_from_strings(
        self, vs: List[List[int]], subset_prefixes=None
    ) -> np.ndarray:
        if subset_prefixes is None:
            subset_prefixes = np.ones(self.num_prefixes, dtype=bool)
        vs_idxs = [self.record_suffix(v)[2] for v in vs]
        return self.compute_decision(vs_idxs, subset_prefixes)

    def compute_decision_array_from_strings(
        self, vs: List[List[int]], subset_prefixes=None
    ) -> np.ndarray:
        decision = self.compute_decision_from_strings(vs, subset_prefixes)
        # NaN (un-evaluated) cells fail both comparisons -> classified as neither
        # accept nor reject, so a partial column excludes those prefixes rather than
        # miscounting them.  Such cells only occur for prefixes outside the predicate's
        # state, which are routed away before the result is used.
        with np.errstate(invalid="ignore"):
            return np.array(
                [
                    decision < self.reject_thresh,
                    decision >= self.accept_thresh,
                ]
            )

    def add_prefixes(self, new_prefixes: List[List[int]]):

        assert new_prefixes, "No new prefixes to add"
        assert len(new_prefixes + self.prefixes) == len(
            set(tuple(p) for p in new_prefixes + self.prefixes)
        ), "Prefixes must be unique"

        # A partial suffix column (some cells NaN) stays partial for new prefixes:
        # those prefixes lie outside the suffix's discovery state, so evaluating them
        # would be exactly the wasted work partial columns exist to avoid.  They are
        # filled lazily by record_suffix if a later round actually needs them.
        partial = [bool(np.isnan(m).any()) for m in self.corresponding_masks]
        additional_prefixes = []
        additional_masks = []
        for prefix in tqdm.tqdm(new_prefixes, desc="Adding new prefixes", delay=1):
            additional_prefixes.append(prefix)
            additional_masks.append(
                np.array(
                    [
                        (
                            np.nan
                            if partial[i]
                            else self.oracle.membership_query(prefix + v)
                        )
                        for i, v in enumerate(self.suffix_bank)
                    ],
                    np.float32,
                )
            )
        additional_masks = np.array(additional_masks).T
        self.prefixes.extend(additional_prefixes)
        # Prefixes added after construction (counterexamples, leaf enrichment)
        # are full-length probe prefixes, hence representative.
        self.representative_prefixes.extend([True] * len(additional_prefixes))

        assert len(self.corresponding_masks) == len(additional_masks)
        self.corresponding_masks = [
            np.concatenate([self.corresponding_masks[i], additional_masks[i]])
            for i in range(len(self.suffix_bank))
        ]

    @property
    def num_prefixes(self) -> int:
        return len(self.prefixes)
