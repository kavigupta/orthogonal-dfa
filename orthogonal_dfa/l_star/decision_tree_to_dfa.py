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

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterator, Union
from automata.fa.dfa import DFA
import numpy as np
import scipy
import tqdm.auto as tqdm

from .sampler import Sampler
from .structures import DecisionTree, Oracle


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


def compute_strings_by_state(
    alphabet_size: int,
    dt: DecisionTree,
    oracle: Oracle,
    sampler: Sampler,
    min_samples_per_state: int,
    *,
    seed: int
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
    seed: int
) -> DFA:
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
            transitions[state, symbol, next_state] += 1
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
    for first, second, _, mask in find_pair_of_correlated_strings(
        for_state, oracle, sampler, p_requirement, attempt_samples_pairs, rng
    ):
        print(mask.mean())
        print("sampled", sum(first), sum(second))
        vs = [first, second]
        for _ in tqdm.trange(attempt_samples, desc="Finding more elements", delay=1):
            v = sampler.sample(alphabet_size=2, rng=rng)
            if v in vs:
                continue
            mask_v = compute_mask(for_state, oracle, v)
            p = chi_squared_p(mask_v, mask)
            # print(sum(v), corr)
            if p < p_requirement:
                vs.append(v)
            if len(vs) >= num_strings:
                return vs
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
                yield v, vs[j], p, prev_mask
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
