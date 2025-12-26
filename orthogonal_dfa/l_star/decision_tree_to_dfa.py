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

from collections import defaultdict
from dataclasses import dataclass
import itertools
from typing import Callable, Dict, List, Optional, Tuple, Iterator, Union
from automata.fa.dfa import DFA
import numpy as np
import scipy
import tqdm.auto as tqdm

from .sampler import Sampler
from .structures import (
    DecisionTree,
    Oracle,
    DecisionTreeInternalNode,
    DecisionTreeLeafNode,
)


@dataclass
class TriPredicate:
    vs: List[List[int]]
    evidence_threshold: float

    def predict(self, x: List[int], oracle: Oracle) -> float:
        return np.mean([oracle.membership_query(x + v) for v in self.vs])

    def __call__(self, x: List[int], oracle: Oracle) -> Union[bool, None]:
        f = self.predict(x, oracle)
        if f > self.evidence_threshold:
            return True
        if f < 1 - self.evidence_threshold:
            return False
        return None

    def __hash__(self):
        return hash((tuple(tuple(v) for v in self.vs), self.evidence_threshold))


def compute_strings_by_state(
    alphabet_size: int,
    dt: DecisionTree,
    oracle: Oracle,
    sampler: Sampler,
    min_samples_per_state: int,
    *,
    seed: int,
) -> Dict[int, List[List[int]]]:
    strings_each = {state: [] for state in range(dt.num_states)}
    rng = np.random.default_rng(seed)
    counts = np.zeros(dt.num_states, dtype=int)
    while np.any(counts < min_samples_per_state):
        string = sampler.sample(rng, alphabet_size)
        state = dt.classify(string, oracle)
        if state is None:
            continue
        strings_each[state].append(string)
        counts[state] += 1
    return strings_each


def decision_tree_to_dfa(
    alphabet_size: int,
    dt: DecisionTree,
    oracle: Oracle,
    sampler: Sampler,
    min_samples_per_state: int,
    *,
    seed: int,
) -> DFA:
    num_states, strings_each, transitions, accepts, counts = (
        compute_decision_tree_stats(
            alphabet_size, dt, oracle, sampler, min_samples_per_state, seed=seed
        )
    )
    transitions = transitions / counts[:, None, None]
    accepts = accepts / counts

    # return transitions, accepts, strings_each
    print(transitions)
    print(accepts)
    transitions = transitions.argmax(axis=2)
    accepts = accepts > 0.5
    possible_dfas = [
        DFA(
            states=set(range(num_states)),
            input_symbols=set(range(alphabet_size)),
            transitions={
                s: {sym: transitions[s, sym] for sym in range(alphabet_size)}
                for s in range(num_states)
            },
            initial_state=initial_state,
            final_states={s for s in range(num_states) if accepts[s]},
        )
        for initial_state in range(num_states)
    ]
    return max(possible_dfas, key=lambda dfa: consistentcy_score(dfa, strings_each))


def compute_decision_tree_stats(
    alphabet_size, dt, oracle, sampler, min_samples_per_state, *, seed
):
    num_states = dt.num_states
    strings_each = {state: [] for state in range(num_states)}
    counts = np.zeros(num_states, dtype=int)
    transitions = np.zeros((num_states, alphabet_size, num_states), dtype=int)
    accepts = np.zeros(num_states, dtype=int)
    rng = np.random.default_rng(seed)
    while np.any(counts < min_samples_per_state):
        string = sampler.sample(rng, alphabet_size)
        state = dt.classify(string, oracle)
        if state is None:
            continue
        strings_each[state].append(string)
        counts[state] += 1
        if oracle.membership_query(string):
            accepts[state] += 1
        for symbol in range(alphabet_size):
            extended_string = string + [symbol]
            next_state = dt.classify(extended_string, oracle)
            if next_state is not None:
                transitions[state, symbol, next_state] += 1
    return num_states, strings_each, transitions, accepts, counts


def consistentcy_score(dfa: DFA, strings_each: Dict[int, List[List[int]]]) -> int:
    score = 0
    for state, strings in strings_each.items():
        for string in strings:
            current_state = dfa.initial_state
            for symbol in string:
                current_state = dfa.transitions[current_state][symbol]
            if current_state == state:
                score += 1
    return score


def compute_corr(a):
    corr = a @ a.T
    return corr


def normalize(a):
    a = a - a.mean()
    a = a / np.linalg.norm(a)
    return a


def find_correlated_strings(
    for_state: List[List[int]],
    oracle: Oracle,
    sampler: Sampler,
    p_requirement: float,
    attempt_samples_pairs: int,
    attempt_samples: int,
    num_strings: int,
) -> Optional[List[List[int]]]:
    rng = np.random.default_rng(0)
    for first, second, _, mask, mask_second in find_pair_of_correlated_strings(
        for_state, oracle, sampler, p_requirement, attempt_samples_pairs, rng
    ):
        print(mask.mean())
        print("sampled", sum(first), sum(second))
        vs = [first, second]
        masks = [mask, mask_second]
        for _ in tqdm.trange(attempt_samples, desc="Finding more elements", delay=1):
            v = sampler.sample(alphabet_size=2, rng=rng)
            if v in vs:
                continue
            mask_v = compute_mask(for_state, oracle, v)
            p = chi_squared_p(mask_v, mask)
            # print(sum(v), corr)
            if p < p_requirement:
                vs.append(v)
                masks.append(mask_v)
            if len(vs) >= num_strings:
                return vs, masks
    return None


def find_pair_of_correlated_strings(
    for_state: List[List[int]],
    oracle: Oracle,
    sampler: Sampler,
    p_requirement: float,
    attempt_samples: int,
    rng: np.random.Generator,
) -> Iterator[Tuple[List[int], List[int], float, np.ndarray]]:
    vs = []
    masks = []
    for _ in tqdm.trange(attempt_samples, desc="Attempting to find pair", delay=1):
        v = sampler.sample(alphabet_size=2, rng=rng)
        if v in vs:
            continue
        mask = compute_mask(for_state, oracle, v)
        for j, prev_mask in enumerate(masks):
            p = chi_squared_p(prev_mask, mask)
            if p < p_requirement:
                print("correlation", np.corrcoef(prev_mask, mask)[0, 1])
                yield v, vs[j], p, prev_mask, mask
        masks.append(mask)
        vs.append(v)


def compute_mask(for_state, oracle, v):
    mask = np.array([oracle.membership_query(x + v) for x in for_state], np.float32)

    return mask


def chi_squared_p(x, y):
    if np.corrcoef(x, y)[0, 1] < 0:
        return 1
    matr = np.zeros((2, 2), dtype=np.int64)
    np.add.at(matr, (x.astype(int), y.astype(int)), 1)
    freqs = matr / matr.sum()
    freqs_x = freqs.sum(0, keepdims=True)
    freqs_y = freqs.sum(1, keepdims=True)
    freqs_expected = freqs_x * freqs_y
    return scipy.stats.chisquare(
        matr.flatten(), freqs_expected.flatten() * matr.sum()
    ).pvalue


def all_correlations(m):
    m -= m.mean(1, keepdims=True)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    return m @ m.T


def best_correlation(m) -> Tuple[int, int, float]:
    m = np.array(m)
    corrs = all_correlations(m)
    np.fill_diagonal(corrs, -1)
    i, j = np.unravel_index(np.argmax(corrs), corrs.shape)
    return i, j, corrs[i, j]


@dataclass
class PrefixSuffixTracker:
    sampler: Sampler
    rng: np.random.Generator
    oracle: Oracle
    alphabet_size: int
    suffix_family_size: int
    chi_squared_p_min: float
    suffix_prevalence: float
    evidence_thresh: float
    prefixes: List[List[int]]
    suffixes_seen: dict
    suffix_bank: List[List[int]]
    corresponding_masks: List[np.ndarray]

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
        chi_squared_p_min: float,
        suffix_prevalence: float,
        evidence_thresh: float,
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
            chi_squared_p_min=chi_squared_p_min,
            suffix_prevalence=suffix_prevalence,
            evidence_thresh=evidence_thresh,
            prefixes=prefixes,
            suffixes_seen={},
            suffix_bank=[],
            corresponding_masks=[],
        )

    def sample_suffix(self) -> Tuple[List[int], np.ndarray, int]:
        while True:
            v = self.sampler.sample(rng=self.rng, alphabet_size=2)
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

    def sample_suffixes(self, num_suffixes: int):
        for _ in tqdm.trange(num_suffixes, desc="Sampling suffixes", delay=1):
            self.sample_suffix()

    def finish_populating_suffix_family(self, vs, limit=None):
        if len(vs) >= self.suffix_family_size:
            return
        pbar = tqdm.tqdm(
            desc="Completing suffix family",
            delay=1,
            total=self.suffix_family_size - len(vs),
        )
        for i in itertools.count():
            if limit is not None and i >= limit:
                pbar.close()
                return
            _, mask, idx = self.sample_suffix()
            if (
                chi_squared_p(self.corresponding_masks[vs[0]], mask)
                < self.chi_squared_p_min
            ):
                pbar.update()
                vs.append(idx)
                if len(vs) >= self.suffix_family_size:
                    pbar.close()
                    return

    def corresponding_masks_for_subset(self, subset_prefixes=None) -> List[np.ndarray]:
        corresponding_masks = np.array(self.corresponding_masks)
        if subset_prefixes is not None:
            corresponding_masks = corresponding_masks[:, subset_prefixes]
        return corresponding_masks

    def best_correlation_in_bank(self, subset_prefixes=None) -> Tuple[int, int, float]:
        corresponding_masks = self.corresponding_masks_for_subset(subset_prefixes)
        idx_1, idx_2, _ = best_correlation(corresponding_masks)
        if (
            chi_squared_p(corresponding_masks[idx_1], corresponding_masks[idx_2])
            < self.chi_squared_p_min
        ):
            return (idx_1, idx_2)
        return None

    def find_suffix_family(
        self, suffix_prevalence_requirement: float, subset_prefixes=None
    ) -> List[int]:
        required_suffix_availability = int(np.ceil(10 / suffix_prevalence_requirement))
        if len(self.suffix_bank) < required_suffix_availability:
            for _ in tqdm.trange(
                required_suffix_availability - len(self.suffix_bank),
                desc="Sampling initial suffixes",
                delay=1,
            ):
                self.sample_suffix()
        best_pair = self.best_correlation_in_bank(subset_prefixes)
        if best_pair is None:
            return None
        vs = self.query_for_mask(
            self.corresponding_masks[best_pair[0]], subset_prefixes
        )
        if len(vs) < suffix_prevalence_requirement * len(self.suffix_bank):
            return None
        self.finish_populating_suffix_family(vs)
        return vs

    def query_for_mask(self, test_mask: np.ndarray, subset_prefixes=None) -> List[int]:
        if subset_prefixes is not None:
            test_mask = test_mask[subset_prefixes]

        corresponding_masks = self.corresponding_masks_for_subset(subset_prefixes)

        return np.where(
            [
                chi_squared_p(test_mask, mask) < self.chi_squared_p_min
                for mask in corresponding_masks
            ]
        )[0].tolist()

    def compute_decision(self, vs, subset_prefixes=None) -> np.ndarray:
        selected_masks = self.corresponding_masks_for_subset(subset_prefixes)[vs]
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

    def extract_decision_tree(self):
        completed_states = []
        states_fringe = [([], None)]
        while states_fringe:
            print(
                f"Expanding: there are {len(states_fringe)} states to expand and {len(completed_states)} completed states"
            )
            path, subset_mask = states_fringe.pop()
            if subset_mask is not None:
                print(f"Expanding mask with {subset_mask.sum()} prefixes")
            vs = self.find_suffix_family(self.suffix_prevalence, subset_mask)
            if vs is None:
                completed_states.append(path)
            else:
                states_fringe.extend(self.split_states(vs, path, subset_mask))
        return flat_decision_tree_to_decision_tree(completed_states)

    def prepend_to_all(self, vs: List[int], prefix: int):
        vs_new = []
        for v in tqdm.tqdm(vs, desc="Prepending to all suffixes", delay=1):
            _, _, v_new = self.record_suffix([prefix] + self.suffix_bank[v])
            vs_new.append(v_new)
        return vs_new

    def execute_path(
        self, path: List[Tuple[TriPredicate, bool]], prepend=()
    ) -> np.ndarray:
        mask = np.ones(len(self.prefixes), dtype=bool)
        for predicate, decision in path:
            vs_idxs = [self.record_suffix([*prepend, *v])[2] for v in predicate.vs]
            decision_mask = self.compute_decision(vs_idxs, subset_prefixes=mask)
            if decision:
                decision_mask = decision_mask >= self.evidence_thresh
            else:
                decision_mask = decision_mask < 1 - self.evidence_thresh
            mask = cascade(mask, decision_mask)
        return mask

    def compute_transition_matrix(
        self, paths: List[List[Tuple[TriPredicate, bool]]]
    ) -> np.ndarray:
        state_map = np.array([self.execute_path(path) for path in paths]).astype(int)
        transitions = []
        for symbol in range(self.alphabet_size):
            pre = np.array(
                [self.execute_path(path, [symbol]) for path in paths]
            ).astype(int)
            transitions.append((pre @ state_map.T).argmax(0))
        transitions = np.array(transitions)
        return transitions.T

    def compute_accepts_vector(
        self, paths: List[List[Tuple[TriPredicate, bool]]]
    ) -> np.ndarray:
        accepts = []
        for ((pred, decision), *_) in paths:
            assert [] in pred.vs
            accepts.append(decision)
        return np.array(accepts)

    def possible_dfas(self, paths: List[List[Tuple[TriPredicate, bool]]]) -> List[DFA]:
        transitions = self.compute_transition_matrix(paths)
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

    def dfa_success_rates(
        self, dfas: List[DFA], paths: List[List[Tuple[TriPredicate, bool]]]
    ) -> List[float]:
        dt = flat_decision_tree_to_decision_tree(paths)
        odfa = [[dfa.accepts_input(string) for string in self.prefixes] for dfa in dfas]
        odfa = np.array(odfa)
        assert (
            [] in dt.predicate.vs
        ), "The root predicate must include the empty string as an exemplar"
        decision = self.compute_decision(
            [self.record_suffix(v)[2] for v in dt.predicate.vs]
        )
        decision_arr = np.array(
            [decision < 1 - self.evidence_thresh, decision >= self.evidence_thresh]
        )
        mask = decision_arr.any(0)
        odfa, decision_arr = odfa[:, mask], decision_arr[:, mask]

        odfa = np.stack([~odfa, odfa], axis=1)
        return ((odfa & decision_arr).sum(-1) / decision_arr.sum(-1)).mean(1)

    def optimal_dfa(self, paths: List[List[Tuple[TriPredicate, bool]]]) -> DFA:
        possible_dfas = self.possible_dfas(paths)
        success_rates = self.dfa_success_rates(possible_dfas, paths)
        best_idx = np.argmax(success_rates)
        print(f"Best DFA has success rate on 'correct' states {success_rates[best_idx]:.4f}")
        return possible_dfas[best_idx]


def cascade(mask_1, mask_2):
    if mask_1 is None:
        return mask_2
    mask_1 = mask_1.copy()
    mask_1[mask_1] = mask_2
    return mask_1


def flat_decision_tree_to_decision_tree(
    fdt: List[List[Tuple[TriPredicate, bool]]],
) -> DecisionTree:
    """
    Takes a flat decision tree (fdt), which is represented as a list of descriptors of leaves, each
    being a list of decisions made along the path from the root to the leaf (represented as a tuple (predicate, decision))),
    and converts it into a hierarchical DecisionTree structure.
    """
    if not fdt:
        raise ValueError("Flat decision tree cannot be empty")

    # let partial_tree be a dictionary mapping from paths to DecisionTree nodes
    partial_tree: Dict[Tuple[Tuple[TriPredicate, bool], ...], DecisionTree] = {
        tuple(path): DecisionTreeLeafNode(i) for i, path in enumerate(fdt)
    }
    while len(partial_tree) > 1:
        # attempt to merge nodes that are the same except for the last decision
        path_1, path_2 = locate_mergeable_paths(partial_tree)
        # print(path_1)
        # print(path_2)
        (*prefix, (predicate, is_accepting)) = path_1
        if is_accepting:
            path_1, path_2 = path_2, path_1
        node = DecisionTreeInternalNode(
            predicate=predicate,
            by_rejection=(
                partial_tree[path_1],
                partial_tree[path_2],
            ),
        )
        del partial_tree[path_1]
        del partial_tree[path_2]
        partial_tree[tuple(prefix)] = node
    return partial_tree[()]


def locate_mergeable_paths(
    partial_tree: Dict[Tuple[Tuple[TriPredicate, bool], ...], DecisionTree],
) -> Tuple[
    Tuple[Tuple[TriPredicate, bool], ...], Tuple[Tuple[TriPredicate, bool], ...]
]:
    by_everything_but_last = defaultdict(list)
    for path in partial_tree.keys():
        by_everything_but_last[path[:-1]].append(path)
    assert any(len(v) >= 2 for v in by_everything_but_last.values())
    prefix = next(p for p, v in by_everything_but_last.items() if len(v) >= 2)
    assert len(by_everything_but_last[prefix]) == 2
    first, second = by_everything_but_last[prefix]
    assert first[-1][0] == second[-1][0]
    assert {first[-1][1], second[-1][1]} == {True, False}
    return first, second


def overlaps(pst, states, vs, *, min_state_size):
    masks = np.array(
        [
            pst.compute_decision(vs) > pst.evidence_thresh,
            pst.compute_decision(vs) < 1 - pst.evidence_thresh,
        ]
    )
    existing_states = np.array([m for _, m in states])
    valid = np.any(masks, 0) & np.any(existing_states, 0)
    masks, existing_states = masks[:, valid], existing_states[:, valid]
    freqs = (masks[:, None] & existing_states[None]).mean(-1)
    return np.where((freqs > min_state_size).all(0))[0].tolist()
