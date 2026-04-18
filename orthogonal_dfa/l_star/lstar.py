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

import numpy as np
import tqdm.auto as tqdm
from automata.fa.dfa import DFA

from .dfa_utils import final_states_all_initial, states_intermediate
from .state_discovery import discover_states
from .statistics import unlikely_this_many_agreements
from .structures import DecisionTree, DecisionTreeLeafNode, TriPredicate


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
                    [[c] + x for x in p.vs], p.accept_threshold, p.reject_threshold
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
    print("Transition matrix:")
    print(transitions)

    return transitions.argmax(-1)


def optimal_dfa(pst, dt: DecisionTree):
    transitions = compute_transition_matrix(pst, dt)
    num_states = dt.num_states

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
    disagreements = np.where((dfa_states[best_idx] != dt_states))[0]
    print(f"Disagreements on indices {disagreements} out of {len(dt_states)}")
    for idx in disagreements:
        print(idx)
        print(
            f"  prefix {pst.prefixes[idx]} classified as {dt_states[idx]} by DT, "
            f"but DFA has final state {dfa_states[best_idx, idx]}"
        )
    return dfas[best_idx]


def add_counterexample_prefixes(pst, dt, dfa, count, *, expected_acc):
    results = generate_counterexamples(
        pst,
        pst.sampler,
        pst.oracle,
        dt,
        dfa,
        count=count,
        expected_acc=expected_acc,
    )
    pst.add_prefixes(results)
    return results


def locate_incorrect_point(oracle, dt, dfa, x, y):
    s0 = dt.classify(x, oracle)
    if s0 is None:
        return None, "could not classify initial state"
    dfa_states_each = states_intermediate(s0, y, dfa)
    if dt.classify(x + y, oracle) == dfa_states_each[-1]:
        return None, "no inconsistency"
    correct_idx = 0
    incorrect_idx = len(y)
    # binary search for first incorrect index
    while correct_idx < incorrect_idx - 1:
        mid_idx = (correct_idx + incorrect_idx) // 2
        dt_state = dt.classify(x + y[: mid_idx + 1], oracle)
        if dt_state is None:
            return None, "could not classify state during binary search"
        if dt_state == dfa_states_each[mid_idx + 1]:
            correct_idx = mid_idx
        else:
            incorrect_idx = mid_idx
    return x + y[: correct_idx + 1], y[correct_idx + 1]


def generate_counterexamples(pst, us, oracle, dt, dfa, *, count, expected_acc):
    boundary = pst.decision_boundary
    # The counterexample pipeline classifies strings many times: ~log2(string_len)
    # binary search steps + 2 decisive checks, each traversing the full DT.  A
    # false positive just adds an uninformative prefix (harmless), so we can
    # tolerate a much higher overall error rate than state discovery (which uses
    # decision_rule_fpr).  We use 0.2 as the whole-pipeline budget and union-bound
    # over all node-level decisions.
    from .statistics import compute_suffix_size_counterexample_gen as _compute_sfx

    counterexample_fpr = 0.2
    string_len = pst.sampler.length
    num_classifications = 2 + int(np.ceil(np.log2(string_len)))
    num_node_decisions = num_classifications * dt.depth
    effective_p = 0.5 + pst.config.min_signal_strength
    per_node_budget = counterexample_fpr / max(num_node_decisions, 1)
    scaled_suffix_size = _compute_sfx(per_node_budget, effective_p)
    dt_with_reduced_predicates = dt.map_over_predicates(
        lambda p: TriPredicate(p.vs[:scaled_suffix_size], boundary, boundary)
    )
    dt_with_decisive_predicates = dt.map_over_predicates(
        lambda p: TriPredicate(p.vs, boundary, boundary)
    )
    pbar = tqdm.tqdm(total=count)
    additional_prefixes = []
    num_samples = 0
    num_agreements = 0
    while True:
        num_samples += 1
        x = us.sample(pst.rng, pst.alphabet_size)
        y = us.sample(pst.rng, pst.alphabet_size)
        prefix, sym = locate_incorrect_point(
            oracle,
            dt_with_reduced_predicates,
            dfa,
            x,
            y,
        )
        if prefix is None:
            num_agreements += sym == "no inconsistency"
            if unlikely_this_many_agreements(num_agreements, num_samples, expected_acc):
                raise AssertionError(
                    f"Observed {num_agreements} 'no inconsistency' results in"
                    f" {num_samples} samples, which is unlikely given expected"
                    f" accuracy {expected_acc:.3f}"
                )
            continue
        if prefix in additional_prefixes or prefix in pst.prefixes:
            continue
        state_1 = dt_with_decisive_predicates.classify(prefix, oracle)
        state_2 = dfa.transitions[state_1][sym]
        if state_2 == dt_with_decisive_predicates.classify(prefix + [sym], oracle):
            continue
        additional_prefixes.append(prefix)
        pbar.update()
        if len(additional_prefixes) >= count:
            pbar.close()
            return additional_prefixes


def estimate_agreement_rate(pst, us, oracle, dt_decisive, dfa, *, num_samples):
    """
    Estimate the DFA's true agreement rate with the DT on fresh random strings,
    starting from the empty prefix (so the DFA simulates from its actual
    initial_state).  Classification failures are excluded from the denominator.
    """
    agreements = 0
    valid = 0
    for _ in range(num_samples):
        y = us.sample(pst.rng, pst.alphabet_size)
        prefix, reason = locate_incorrect_point(oracle, dt_decisive, dfa, [], y)
        if prefix is None and reason == "no inconsistency":
            agreements += 1
            valid += 1
        elif prefix is not None:
            valid += 1
    return agreements / valid if valid else 0.0


def counterexample_driven_synthesis(
    pst, *, additional_counterexamples: int, acc_threshold: float
):
    first_round = True
    while True:
        print(f"Starting synthesis iteration with {pst.num_prefixes} prefixes")
        while True:
            dt = discover_states(pst, first_round=first_round)
            first_round = False
            print(f"Extracted flat decision tree with {dt.num_states} states")
            if dt.num_states > 1:
                break
            pst.sample_more_prefixes()
        dfa = optimal_dfa(pst, dt)
        print("DFA found!")
        print(dfa)
        boundary = pst.decision_boundary
        dt_decisive = dt.map_over_predicates(
            lambda p: TriPredicate(p.vs, boundary, boundary)
        )
        true_acc = estimate_agreement_rate(
            pst, pst.sampler, pst.oracle, dt_decisive, dfa, num_samples=500
        )
        print(f"Estimated DFA accuracy on fresh samples: {true_acc:.4f}")
        if true_acc >= acc_threshold:
            print(f"Achieved desired accuracy of {acc_threshold}; stopping synthesis")
            yield dfa, dt, None
            return
        add_counterexample_prefixes(
            pst, dt, dfa, additional_counterexamples, expected_acc=acc_threshold
        )
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
