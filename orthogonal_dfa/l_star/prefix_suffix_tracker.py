from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import tqdm.auto as tqdm

from .sampler import Sampler
from .structures import Oracle


def compute_mask(for_state, oracle, v):
    mask = np.array([oracle.membership_query(x + v) for x in for_state], np.float32)

    return mask


@dataclass
class SearchConfig:
    suffix_family_size: int
    evidence_thresh: float
    decision_rule_fpr: float
    suffix_size_counterexample_gen: int
    num_addtl_prefixes: Optional[int] = None
    fnr_limit: float = 0.02
    split_pval: float = 0.001


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

    @property
    def alphabet_size(self) -> int:
        return self.oracle.alphabet_size

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
        prefixes = [
            sampler.sample(rng, alphabet_size=oracle.alphabet_size)
            for _ in range(num_prefixes)
        ]
        return cls(
            sampler=sampler,
            rng=rng,
            oracle=oracle,
            config=config,
            prefixes=prefixes,
            suffixes_seen={},
            suffix_bank=[],
            corresponding_masks=[],
        )

    def sample_suffix(self) -> Tuple[List[int], np.ndarray, int]:
        while True:
            v = self.sampler.sample(rng=self.rng, alphabet_size=self.alphabet_size)
            if tuple(v) in self.suffixes_seen:
                continue
            return self.record_suffix(v)

    def record_suffix(self, v: List[int]) -> Tuple[List[int], np.ndarray, int]:
        if tuple(v) in self.suffixes_seen:
            return self.suffixes_seen[tuple(v)]
        self.suffix_bank.append(v)
        mask = compute_mask(self.prefixes, self.oracle, v)
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
        """
        arr = self.compute_decision_array_from_strings(
            [self.suffix_bank[v] for v in vs]
        ).mean(1)
        if arr.min() == 0:
            return 1
        return 1 - arr.sum()

    def sample_more_prefixes(self):
        # Sample random prefixes and add them
        new_prefixes = [
            self.sampler.sample(self.rng, alphabet_size=self.alphabet_size)
            for _ in range(self.config.num_addtl_prefixes)
        ]
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

    def compute_decision_from_strings(self, vs: List[List[int]]) -> np.ndarray:
        vs_idxs = [self.record_suffix(v)[2] for v in vs]
        return self.compute_decision(vs_idxs, np.ones(self.num_prefixes, dtype=bool))

    def compute_decision_array_from_strings(self, vs: List[List[int]]) -> np.ndarray:
        decision = self.compute_decision_from_strings(vs)
        return np.array(
            [
                decision < 1 - self.config.evidence_thresh,
                decision >= self.config.evidence_thresh,
            ]
        )

    def add_prefixes(self, new_prefixes: List[List[int]]):

        assert new_prefixes, "No new prefixes to add"

        additional_prefixes = []
        additional_masks = []
        for prefix in tqdm.tqdm(new_prefixes, desc="Adding new prefixes", delay=1):
            if prefix not in self.prefixes:
                additional_prefixes.append(prefix)
                additional_masks.append(
                    np.array(
                        [
                            self.oracle.membership_query(prefix + v)
                            for v in self.suffix_bank
                        ],
                        np.float32,
                    )
                )
        additional_masks = np.array(additional_masks).T
        self.prefixes.extend(additional_prefixes)

        assert len(self.corresponding_masks) == len(additional_masks)
        self.corresponding_masks = [
            np.concatenate([self.corresponding_masks[i], additional_masks[i]])
            for i in range(len(self.suffix_bank))
        ]

    @property
    def num_prefixes(self) -> int:
        return len(self.prefixes)
