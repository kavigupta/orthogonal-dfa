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
    states_after_c_list = [
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

    for c, states_c in enumerate(states_after_c_list):
        valid = states_c >= 0
        np.add.at(
            transitions,
            (states[valid], c, states_c[valid]),
            1,
        )

    # Pick the best target for each (source, symbol) pair
    # If no confident votes, use unconfident votes as tiebreaker
    result = np.zeros((num_states, pst.alphabet_size), dtype=int)
    for src in range(num_states):
        for sym in range(pst.alphabet_size):
            confident_votes = transitions[src, sym, :]

            if confident_votes.max() > 0:
                # We have confident votes; use them
                result[src, sym] = np.argmax(confident_votes)
            else:
                # No confident votes for this transition
                # Count unconfident observations: where does (src, sym) go
                # when the target state is unconfident?
                states_c = states_after_c_list[sym]
                mask = (states == src) & (states_c < 0)

                if mask.any():
                    # We have unconfident observations
                    # Look at some heuristic: e.g., which state is reachable from src?
                    # For now, pick the most common confident target from sym,
                    # or the first state if all are unconfident
                    # Actually, just pick argmax of all transitions for this symbol (best guess)
                    all_targets_for_sym = transitions[src, sym, :]
                    if all_targets_for_sym.max() > 0:
                        result[src, sym] = np.argmax(all_targets_for_sym)
                    else:
                        # Truly no data; pick lowest index
                        result[src, sym] = 0
                else:
                    # No observations at all for this transition
                    result[src, sym] = 0

    return result


def _check_and_enrich_insufficient_states(pst, dt, min_prefixes=30):
    """Check if any state has insufficient confident votes for its transitions.

    If a state has too few observations, sample more prefixes and add them to the PST.

    Args:
        pst: PrefixSuffixTracker
        dt: DecisionTree
        min_prefixes: Minimum number of prefixes per state (default 30)

    Returns:
        bool: True if enrichment was needed and performed, False otherwise
    """
    # Count how many confident prefixes reach each state
    dt_states = classify_states_with_decision_tree(pst, dt)
    confident = dt_states >= 0

    state_counts = np.bincount(
        dt_states[confident], minlength=dt.num_states
    )

    # Find states with insufficient prefixes
    insufficient = state_counts < min_prefixes
    insufficient_states = np.where(insufficient)[0]

    if len(insufficient_states) == 0:
        return False

    print(
        f"State enrichment: states {insufficient_states.tolist()} have "
        f"<{min_prefixes} prefixes. Sampling more..."
    )

    # Sample random prefixes and add those that reach insufficient states
    new_prefixes = []
    us = pst.sampler
    oracle = pst.oracle
    total_needed = (min_prefixes - state_counts[insufficient_states]).sum()
    found = 0

    # Try up to 10x the needed amount
    for _ in range(total_needed * 10):
        if found >= total_needed:
            break

        # Sample a random prefix
        prefix = us.sample(pst.rng, pst.alphabet_size)

        if prefix not in pst.prefixes:
            # Classify it through the DT
            prefix_state = dt.classify(prefix, oracle)
            if prefix_state in insufficient_states:
                new_prefixes.append(prefix)
                found += 1

    if new_prefixes:
        print(f"Found {len(new_prefixes)} new prefixes via sampling")
        pst.add_prefixes(new_prefixes)
        return True

    return False


def optimal_dfa(pst, dt: DecisionTree):
    # Check if any state has insufficient data; if so, enrich and retry
    max_enrichment_rounds = 3
    for _ in range(max_enrichment_rounds):
        if not _check_and_enrich_insufficient_states(pst, dt, min_prefixes=30):
            break

    # Compute transition matrix with enriched data
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
    return success_rates[best_idx], dfas[best_idx]


def add_counterexample_prefixes(pst, dt, dfa, count):
    results = generate_counterexamples(
        pst,
        pst.sampler,
        pst.oracle,
        dt,
        dfa,
        count=count,
    )
    pst.add_prefixes(results)
    return results


def locate_incorrect_point(oracle, dt, dfa, x, y):
    s0 = dt.classify(x, oracle)
    if s0 is None:
        return None
    dfa_states_each = states_intermediate(s0, y, dfa)
    if dt.classify(x + y, oracle) == dfa_states_each[-1]:
        return None
    correct_idx = 0
    incorrect_idx = len(y)
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


def generate_counterexamples(pst, us, oracle, dt, dfa, *, count):
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
    while True:
        y = us.sample(pst.rng, pst.alphabet_size)
        # Start from the empty string so the DT and DFA agree on the
        # initial state (both use dfa.initial_state).  Using a random x
        # causes problems when the DT and DFA disagree on x's state,
        # which corrupts the DFA path used for comparison.
        prefix_and_sym = locate_incorrect_point(
            oracle,
            dt_with_reduced_predicates,
            dfa,
            [],
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
        acc, dfa = optimal_dfa(pst, dt)
        print("DFA found!")
        print(dfa)
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
