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
from typing import List

import numpy as np
import scipy
import tqdm.auto as tqdm
from automata.fa.dfa import DFA

from .dfa_utils import final_states_all_initial, states_intermediate
from .structures import (
    DecisionTree,
    DecisionTreeLeafNode,
    TriPredicate,
    flat_decision_tree_to_decision_tree,
)
from .cluster import sample_suffix_family


def cascade(mask_1, mask_2):
    mask_1 = mask_1.copy()
    mask_1[mask_1] = mask_2
    return mask_1


def split_states(pst, vs, path, subset_mask):
    decision = pst.compute_decision(vs, subset_mask)
    vs_actual = [pst.suffix_bank[v] for v in vs]
    return [
        (
            path + [(TriPredicate(vs_actual, pst.config.evidence_thresh), True)],
            cascade(subset_mask, decision >= pst.config.evidence_thresh),
        ),
        (
            path + [(TriPredicate(vs_actual, pst.config.evidence_thresh), False)],
            cascade(subset_mask, decision < 1 - pst.config.evidence_thresh),
        ),
    ]


def prepend_to_all(pst, vs: List[int], prefix: int):
    vs_new = []
    for v in tqdm.tqdm(vs, desc="Prepending to all suffixes", delay=1):
        _, _, v_new = pst.record_suffix([prefix] + pst.suffix_bank[v])
        vs_new.append(v_new)
    return vs_new


def classify_states_with_decision_tree(pst, dt: DecisionTree):
    if isinstance(dt, DecisionTreeLeafNode):
        return np.full(len(pst.prefixes), dt.state_idx)
    results = np.full(len(pst.prefixes), -1)
    rej, acc = pst.compute_decision_array_from_strings(dt.predicate.vs)
    results[rej] = classify_states_with_decision_tree(pst, dt.by_rejection[0])[rej]
    results[acc] = classify_states_with_decision_tree(pst, dt.by_rejection[1])[acc]
    return results


def compute_transition_matrix(pst, dt: DecisionTree) -> np.ndarray:
    states = classify_states_with_decision_tree(pst, dt)
    states_after_c = [
        classify_states_with_decision_tree(
            pst,
            dt.map_over_predicates(
                lambda p, c=c: TriPredicate(
                    [[c] + x for x in p.vs], p.evidence_threshold
                )
            ),
        )
        for c in range(pst.alphabet_size)
    ]
    num_states = dt.num_states
    transitions = np.zeros((num_states, pst.alphabet_size, num_states), dtype=int)
    for c, states_c in enumerate(states_after_c):
        valid = states_c >= 0
        np.add.at(
            transitions,
            (states[valid], c, states_c[valid]),
            1,
        )
    return transitions.argmax(-1)


def optimal_dfa(pst, paths):
    dt = flat_decision_tree_to_decision_tree(paths)
    transitions = compute_transition_matrix(pst, dt)
    num_states = len(paths)

    accepting_states = set(dt.by_rejection[1].collect_states())

    dfas = [
        DFA(
            states=set(range(num_states)),
            input_symbols=set(range(pst.alphabet_size)),
            transitions={
                s: {sym: transitions[s, sym] for sym in range(pst.alphabet_size)}
                for s in range(num_states)
            },
            initial_state=initial_state,
            final_states=accepting_states,
        )
        for initial_state in range(num_states)
    ]

    # Compare DFA state assignments against decision tree state assignments
    dt_states = classify_states_with_decision_tree(pst, dt)
    confident = dt_states >= 0
    dt_states = dt_states[confident]

    dfa_states = final_states_all_initial(
        transitions, [pre for pre, is_conf in zip(pst.prefixes, confident) if is_conf]
    )
    success_rates = (dfa_states == dt_states).mean(1)

    best_idx = np.argmax(success_rates)
    print(
        f"Best DFA has success rate on 'correct' states {success_rates[best_idx]:.4f}"
    )
    return success_rates[best_idx], dfas[best_idx]


def add_counterexample_prefixes(pst, dt, dfa, count):
    results = generate_counterexamples(
        pst,
        pst.sampler,
        pst.oracle,
        dt,
        dfa,
        count=count,
        suffix_size_counterexample_gen=pst.config.suffix_size_counterexample_gen,
    )
    pst.add_prefixes(results)
    return results


def overlaps(pst, states, vs):
    decision = pst.compute_decision(vs, np.ones(pst.num_prefixes, dtype=bool))
    masks = np.array(
        [
            decision > pst.config.evidence_thresh,
            decision < 1 - pst.config.evidence_thresh,
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
            1 - scipy.stats.binom.cdf(n1, denom, pst.config.decision_rule_fpr),
            1 - scipy.stats.binom.cdf(n2, denom, pst.config.decision_rule_fpr),
        ]
        pval = max(pvals)
        if pval < pst.config.split_pval:
            split_idxs.append(i)
    return split_idxs


def abstract_interpretation_algorithm(pst) -> List[DecisionTree]:
    _, _, v_idx = pst.record_suffix([])
    vs = sample_suffix_family(pst, v_idx)
    vs_queue = [([], vs)]
    states = [([], np.ones(len(pst.prefixes), bool))]

    def split_with(state_indices, vs):
        states_to_split = [states.pop(i) for i in reversed(sorted(state_indices))]
        for decision, m2 in states_to_split:
            states.extend(split_states(pst, vs, decision, m2))

    while vs_queue:
        path, vs_current = vs_queue.pop()
        print(f"Num states: {len(states)}; processing {path}")
        ol = overlaps(pst, states, vs_current)
        if not ol:
            print("Done")
            continue
        split_with(ol, vs_current)
        vs_queue.extend(
            ([c] + path, prepend_to_all(pst, vs_current, c))
            for c in range(pst.alphabet_size)
        )

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
        additional_prefixes.append(prefix)
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
        acc, dfa = optimal_dfa(pst, fdt)
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
        add_counterexample_prefixes(pst, dt, dfa, additional_counterexamples)
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
