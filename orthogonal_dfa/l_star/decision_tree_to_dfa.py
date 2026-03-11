"""
Key Challenges:

Splitting on a criterion does not exclude the possibility that the same criterion could come up again. We need
a way to ensure that if a decision is made, the same decision will not be made later. Possible fix: require
a full set of classifier strings. Not sure why this would work, but maybe it will.

Maybe one thing we could do is have "confident" classifications during the creation, like just drop everything
in the classification between 40% and 60%. This way, we have much greater confidence that we won't find the same
thing twice, and therefore have lower thresholds otherwise.

Things to work on:

Evidence thresholds need some work. Currently there's the possibiliy of p-hacking. We need to do multiple comparisons.


"""

import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import scipy
import tqdm.auto as tqdm
from automata.fa.dfa import DFA

from .dfa_utils import states_intermediate
from .sampler import Sampler
from .structures import (
    DecisionTree,
    DecisionTreeLeafNode,
    Oracle,
    TriPredicate,
    flat_decision_tree_to_decision_tree,
)


def compute_mask(for_state, oracle, v):
    mask = np.array([oracle.membership_query(x + v) for x in for_state], np.float32)

    return mask


@dataclass
class PrefixSuffixTracker:
    sampler: Sampler
    rng: np.random.Generator
    oracle: Oracle
    alphabet_size: int
    suffix_family_size: int
    evidence_thresh: float
    prefixes: List[List[int]]
    suffixes_seen: dict
    suffix_bank: List[List[int]]
    corresponding_masks: List[np.ndarray]
    suffix_size_counterexample_gen: int
    decision_rule_fpr: float
    fnr_limit: float = 0.02
    split_pval: float = 0.001
    p_value_accept: float = 0.01
    num_addtl_prefixes: Optional[int] = None

    @classmethod
    def create(
        cls,
        sampler,
        rng,
        oracle,
        *,
        alphabet_size: int,
        num_prefixes: int,
        suffix_family_size: int,
        evidence_thresh: float,
        decision_rule_fpr: float,
        suffix_size_counterexample_gen,
        num_addtl_prefixes: Optional[int] = None,
    ) -> "PrefixSuffixTracker":
        prefixes = [
            sampler.sample(rng, alphabet_size=alphabet_size)
            for _ in range(num_prefixes)
        ]
        return cls(
            sampler=sampler,
            rng=rng,
            oracle=oracle,
            alphabet_size=alphabet_size,
            suffix_family_size=suffix_family_size,
            evidence_thresh=evidence_thresh,
            prefixes=prefixes,
            suffixes_seen={},
            suffix_bank=[],
            corresponding_masks=[],
            decision_rule_fpr=decision_rule_fpr,
            num_addtl_prefixes=num_addtl_prefixes,
            suffix_size_counterexample_gen=suffix_size_counterexample_gen,
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

    def finish_populating_suffix_family_without_sampling(
        self, vs: List[int], suffix_family_size: int
    ) -> List[int]:
        masks = np.array(self.corresponding_masks)
        cluster = vs
        loss = float("inf")
        while True:
            cluster_center = masks[cluster].mean(0) > 0.5
            losses = np.abs(masks - cluster_center).sum(1)
            cluster = losses.argsort()[: suffix_family_size - len(vs)]
            new_loss = losses[cluster].sum()
            if new_loss >= loss:
                break
            loss = new_loss
        vs += cluster.tolist()

    def sample_suffix_family(self, v: int) -> List[int]:
        prev_fnr = 1.0  # default start with a large value
        strategy = "suffix"
        while True:

            vs = [v]
            self.finish_populating_suffix_family_without_sampling(
                vs, self.suffix_family_size
            )

            fnr = 1 if len(vs) < self.suffix_family_size else self.compute_fnr(vs)
            if fnr <= self.fnr_limit:
                print("FNR limit reached")
                return vs

            # always switch strategies if on prefix mode, because it is way slower, so we should give
            # suffixes a chance.
            if fnr >= prev_fnr or strategy == "prefix":  # switch strategy
                strategy = "prefix" if strategy == "suffix" else "suffix"

            prev_fnr = fnr

            print(f"FNR {fnr:.4f} too high, sampling more suffixes")

            if strategy == "suffix":
                self.sample_more_suffixes(amount=self.suffix_family_size)
            else:
                self.sample_more_prefixes()

    def sample_more_prefixes(self):
        # Sample random prefixes and add them
        new_prefixes = [
            self.sampler.sample(self.rng, alphabet_size=self.alphabet_size)
            for _ in range(self.num_addtl_prefixes)
        ]
        self.add_prefixes(new_prefixes)

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
        return 1 - arr.sum(0)

    def sample_more_suffixes(self, *, amount: int):
        for _ in tqdm.trange(amount, desc="Completing suffix family", delay=1):
            self.sample_suffix()

    def corresponding_masks_for_subset(self, subset_prefixes=None) -> List[np.ndarray]:
        corresponding_masks = np.array(self.corresponding_masks)
        if subset_prefixes is not None:
            corresponding_masks = corresponding_masks[:, subset_prefixes]
        return corresponding_masks

    def compute_decision(self, vs, subset_prefixes=None) -> np.ndarray:
        selected_masks = self.corresponding_masks_for_subset(subset_prefixes)[vs]
        if subset_prefixes is None:
            assert (
                selected_masks.shape[1] == self.num_prefixes
            ), f"Expected {self.num_prefixes}, got {selected_masks.shape[1]}"
        else:
            assert selected_masks.shape[1] == sum(
                subset_prefixes
            ), f"Expected {sum(subset_prefixes)}, got {selected_masks.shape[1]}"
        return selected_masks.mean(0)

    def split_states(
        self,
        vs: List[int],
        path: List[Tuple[TriPredicate, bool]],
        subset_mask: Optional[np.ndarray],
    ) -> list:
        decision = self.compute_decision(vs, subset_mask)
        vs_actual = [self.suffix_bank[v] for v in vs]
        return [
            (
                path + [(TriPredicate(vs_actual, self.evidence_thresh), True)],
                cascade(subset_mask, decision >= self.evidence_thresh),
            ),
            (
                path + [(TriPredicate(vs_actual, self.evidence_thresh), False)],
                cascade(subset_mask, decision < 1 - self.evidence_thresh),
            ),
        ]

    def prepend_to_all(self, vs: List[int], prefix: int):
        vs_new = []
        for v in tqdm.tqdm(vs, desc="Prepending to all suffixes", delay=1):
            _, _, v_new = self.record_suffix([prefix] + self.suffix_bank[v])
            vs_new.append(v_new)
        return vs_new

    def compute_transition_matrix(self, dt: DecisionTree) -> np.ndarray:
        states = self.classify_states_with_decision_tree(dt)
        states_after_c = [
            self.classify_states_with_decision_tree(
                dt.map_over_predicates(
                    lambda p, c=c: TriPredicate(
                        [[c] + x for x in p.vs], p.evidence_threshold
                    )
                )
            )
            for c in range(self.alphabet_size)
        ]
        num_states = dt.num_states
        transitions = np.zeros((num_states, self.alphabet_size, num_states), dtype=int)
        for c, states_c in enumerate(states_after_c):
            valid = states_c >= 0
            np.add.at(
                transitions,
                (states[valid], c, states_c[valid]),
                1,
            )
        return transitions.argmax(-1)

    def compute_accepts_vector(
        self, paths: List[List[Tuple[TriPredicate, bool]]]
    ) -> np.ndarray:
        accepts = []
        for (pred, decision), *_ in paths:
            assert [] in pred.vs
            accepts.append(decision)
        return np.array(accepts)

    def possible_dfas(self, paths: List[List[Tuple[TriPredicate, bool]]]) -> List[DFA]:
        dt = flat_decision_tree_to_decision_tree(paths)
        transitions = self.compute_transition_matrix(dt)
        accepts = self.compute_accepts_vector(paths)
        num_states = len(paths)
        possible_dfas = [
            DFA(
                states=set(range(num_states)),
                input_symbols=set(range(self.alphabet_size)),
                transitions={
                    s: {sym: transitions[s, sym] for sym in range(self.alphabet_size)}
                    for s in range(num_states)
                },
                initial_state=initial_state,
                final_states={s for s in range(num_states) if accepts[s]},
            )
            for initial_state in range(num_states)
        ]
        return possible_dfas

    def compute_decision_from_strings(self, vs: List[List[int]]) -> np.ndarray:
        vs_idxs = [self.record_suffix(v)[2] for v in vs]
        return self.compute_decision(vs_idxs)

    def compute_decision_array_from_strings(self, vs: List[List[int]]) -> np.ndarray:
        decision = self.compute_decision_from_strings(vs)
        return np.array(
            [decision < 1 - self.evidence_thresh, decision >= self.evidence_thresh]
        )

    def classify_states_with_decision_tree(self, dt: DecisionTree):
        if isinstance(dt, DecisionTreeLeafNode):
            return np.full(len(self.prefixes), dt.state_idx)
        results = np.full(len(self.prefixes), -1)
        rej, acc = self.compute_decision_array_from_strings(dt.predicate.vs)
        results[rej] = self.classify_states_with_decision_tree(dt.by_rejection[0])[rej]
        results[acc] = self.classify_states_with_decision_tree(dt.by_rejection[1])[acc]
        return results

    def dfa_success_rates(self, dfas: List[DFA], dt: DecisionTree) -> List[float]:
        odfa = [[dfa.accepts_input(string) for string in self.prefixes] for dfa in dfas]
        odfa = np.array(odfa)
        assert (
            [] in dt.predicate.vs
        ), "The root predicate must include the empty string as an exemplar"
        decision_arr = self.compute_decision_array_from_strings(dt.predicate.vs)
        mask = decision_arr.any(0)
        odfa, decision_arr = odfa[:, mask], decision_arr[:, mask]

        odfa = np.stack([~odfa, odfa], axis=1)
        return ((odfa & decision_arr).sum(-1) / decision_arr.sum(-1)).mean(1)

    def optimal_dfa(self, paths: List[List[Tuple[TriPredicate, bool]]]) -> DFA:
        possible_dfas = self.possible_dfas(paths)
        success_rates = self.dfa_success_rates(
            possible_dfas, flat_decision_tree_to_decision_tree(paths)
        )
        best_idx = np.argmax(success_rates)
        print(
            f"Best DFA has success rate on 'correct' states {success_rates[best_idx]:.4f}"
        )
        return success_rates[best_idx], possible_dfas[best_idx]

    def add_prefixes(self, new_prefixes: List[List[int]]):

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
        if additional_masks:
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

    def add_counterexample_prefixes(self, dt, dfa, count):
        results = generate_counterexamples(
            self,
            self.sampler,
            self.oracle,
            dt,
            dfa,
            count=count,
            suffix_size_counterexample_gen=self.suffix_size_counterexample_gen,
        )
        self.add_prefixes([prefix for prefix, _ in results])
        return results


def cascade(mask_1, mask_2):
    if mask_1 is None:
        return mask_2
    mask_1 = mask_1.copy()
    mask_1[mask_1] = mask_2
    return mask_1


def overlaps(pst, states, vs):
    masks = np.array(
        [
            pst.compute_decision(vs) > pst.evidence_thresh,
            pst.compute_decision(vs) < 1 - pst.evidence_thresh,
        ]
    )
    existing_states = np.array([m for _, m in states])
    assert (
        existing_states.shape[1] == pst.num_prefixes
    ), f"[existing states] Expected {pst.num_prefixes}, got {existing_states.shape[1]}"
    assert (
        masks.shape[1] == pst.num_prefixes
    ), f"[masks] Expected {pst.num_prefixes}, got {masks.shape[1]}"
    valid = np.any(masks, 0) & np.any(existing_states, 0)
    masks, existing_states = masks[:, valid], existing_states[:, valid]
    freqs = (masks[:, None] & existing_states[None]).sum(-1).T
    denominators = freqs.sum(-1)
    print(freqs)
    split_idxs = []
    for i, (denom, (n1, n2)) in enumerate(zip(denominators, freqs)):
        if denom == 0:
            continue
        pvals = [
            1 - scipy.stats.binom.cdf(n1, denom, pst.decision_rule_fpr),
            1 - scipy.stats.binom.cdf(n2, denom, pst.decision_rule_fpr),
        ]
        pval = max(pvals)
        if pval < pst.split_pval:
            split_idxs.append(i)
    return split_idxs


def abstract_interpretation_algorithm(pst) -> List[DecisionTree]:
    _, _, v_idx = pst.record_suffix([])
    vs = pst.sample_suffix_family(v_idx)
    vs_queue = [([], vs)]
    states = [([], np.ones(len(pst.prefixes), bool))]

    def split_with(state_indices, vs):
        states_to_split = [states.pop(i) for i in reversed(sorted(state_indices))]
        for decision, m2 in states_to_split:
            states.extend(pst.split_states(vs, decision, m2))

    while vs_queue:
        path, vs_current = vs_queue.pop()
        print(f"Num states: {len(states)}; processing {path}")
        ol = overlaps(pst, states, vs_current)
        if not ol:
            print("Done")
            continue
        split_with(ol, vs_current)
        if len(states) > 1000:
            raise RuntimeError(
                f"abstract_interpretation_algorithm: state count exploded to {len(states)}"
            )
        vs_queue.extend(
            ([c] + path, pst.prepend_to_all(vs_current, c))
            for c in range(pst.alphabet_size)
        )

    # split_with([0], vs)
    # split_with([0, 1], vs_with_1)
    fdt = [x for x, _ in states]
    return fdt


def locate_incorrect_point(oracle, dt, dfa, x, y):
    s0 = dt.classify(x, oracle)
    if s0 is None:
        return None
    dfa_states_each = states_intermediate(s0, y, dfa)
    if dt.classify(x + y, oracle) == dfa_states_each[-1]:
        return None
    correct_idx = 0
    incorrect_idx = len(x)
    # binary search for first incorrect index
    while correct_idx < incorrect_idx - 1:
        mid_idx = (correct_idx + incorrect_idx) // 2
        dt_state = dt.classify(x + y[: mid_idx + 1], oracle)
        if dt_state is None:
            return None
        if dt_state == dfa_states_each[mid_idx + 1]:
            correct_idx = mid_idx
        else:
            incorrect_idx = mid_idx
    return x + y[: correct_idx + 1], y[correct_idx + 1]


def generate_counterexamples(
    pst, us, oracle, dt, dfa, *, count, suffix_size_counterexample_gen
):
    dt_with_reduced_predicates = dt.map_over_predicates(
        lambda p: TriPredicate(p.vs[:suffix_size_counterexample_gen], 0.5)
    )
    dt_with_decisive_predicates = dt.map_over_predicates(
        lambda p: TriPredicate(p.vs, 0.5)
    )
    pbar = tqdm.tqdm(total=count)
    additional_prefixes = []
    while True:
        x = us.sample(pst.rng, pst.alphabet_size)
        y = us.sample(pst.rng, pst.alphabet_size)
        prefix_and_sym = locate_incorrect_point(
            oracle,
            dt_with_reduced_predicates,
            dfa,
            x,
            y,
        )
        if prefix_and_sym is None:
            continue
        prefix, sym = prefix_and_sym
        state_1 = dt_with_decisive_predicates.classify(prefix, oracle)
        state_2 = dfa.transitions[state_1][sym]
        if state_2 == dt_with_decisive_predicates.classify(prefix + [sym], oracle):
            continue
        additional_prefixes.append(prefix_and_sym)
        pbar.update()
        if len(additional_prefixes) >= count:
            pbar.close()
            return additional_prefixes


def counterexample_driven_synthesis(
    pst, *, additional_counterexamples: int, acc_threshold: float
):
    prev_dfas = []
    while True:
        print(f"Starting synthesis iteration with {pst.num_prefixes} prefixes")
        while True:
            fdt = abstract_interpretation_algorithm(pst)
            print(f"Extracted flat decision tree with {len(fdt)} states")
            if len(fdt) > 1:
                break
            pst.sample_more_prefixes()
        dt = flat_decision_tree_to_decision_tree(fdt)
        acc, dfa = pst.optimal_dfa(fdt)
        print("DFA found!")
        print(dfa)
        if any(
            dfa.issubset(prev_dfa) and prev_dfa.issubset(dfa) for prev_dfa in prev_dfas
        ):
            print("Same DFA twice; stopping synthesis")
            yield dfa, dt, None
            return
        if acc >= acc_threshold:
            print(f"Achieved desired accuracy of {acc_threshold}; stopping synthesis")
            yield dfa, dt, None
            return
        pst.add_counterexample_prefixes(dt, dfa, additional_counterexamples)
        yield dfa, dt, copy.deepcopy(pst)


def do_counterexample_driven_synthesis(
    pst, *, additional_counterexamples: int, acc_threshold: float
) -> DFA:
    dfa = dt = None
    for dfa, dt, _ in counterexample_driven_synthesis(
        pst,
        additional_counterexamples=additional_counterexamples,
        acc_threshold=acc_threshold,
    ):
        pass
    return dfa, dt
