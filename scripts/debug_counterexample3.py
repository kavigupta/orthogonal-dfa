"""Investigate why internal DT-vs-DFA metric is 93.8% when true acc is 99.2%."""

import numpy as np
import time

from orthogonal_dfa.l_star.examples.benchmark_generator import (
    DFAOracle,
    sample_balanced_benchmark,
)
from orthogonal_dfa.l_star.lstar import (
    optimal_dfa,
    add_counterexample_prefixes,
    classify_states_with_decision_tree,
    compute_transition_matrix,
)
from orthogonal_dfa.l_star.state_discovery import discover_states
from orthogonal_dfa.l_star.structures import TriPredicate, SymmetricBernoulli
from orthogonal_dfa.l_star.dfa_utils import final_states_all_initial, states_intermediate
from tests.test_lstar import compute_dfa_accuracy, compute_pst

outer, inner, sep = sample_balanced_benchmark(
    1, alphabet_size=2, num_inner_states=12, num_outer_states=10,
    probe_length=40, min_accept_or_reject=0.15,
)

oracle_creator = lambda nm, s, _dfa=outer: DFAOracle(nm, s, _dfa)
pst = compute_pst(oracle_creator, 0.3, 0)

# Round 0
while True:
    dt = discover_states(pst, first_round=True)
    if dt.num_states > 1:
        break
    pst.sample_more_prefixes()
_, dfa = optimal_dfa(pst, dt)
add_counterexample_prefixes(pst, dt, dfa, 200)

# Round 1
while True:
    dt = discover_states(pst, first_round=False)
    if dt.num_states > 1:
        break
    pst.sample_more_prefixes()
_, dfa = optimal_dfa(pst, dt)
add_counterexample_prefixes(pst, dt, dfa, 200)

# Round 2
while True:
    dt = discover_states(pst, first_round=False)
    if dt.num_states > 1:
        break
    pst.sample_more_prefixes()

print(f"=== Round 2 DT: {dt.num_states} states, depth={dt.depth} ===")

# Reproduce what optimal_dfa does internally
transitions = compute_transition_matrix(pst, dt)
dt_states = classify_states_with_decision_tree(pst, dt)
confident = dt_states >= 0
dt_states_conf = dt_states[confident]

num_states = dt.num_states
dfa_states_all = final_states_all_initial(
    transitions, [pre for pre, is_conf in zip(pst.prefixes, confident) if is_conf]
)

# Find best initial state
success_rates = (dfa_states_all == dt_states_conf).mean(1)
best_idx = np.argmax(success_rates)
print(f"Best initial state: {best_idx}, success_rate: {success_rates[best_idx]:.4f}")
print(f"Total prefixes: {len(pst.prefixes)}, confident: {confident.sum()}, unconfident: {(~confident).sum()}")
print(f"Fraction confident: {confident.mean():.4f}")

# Where do they disagree?
dfa_states_best = dfa_states_all[best_idx]
disagree = dfa_states_best != dt_states_conf
print(f"\nDisagreements: {disagree.sum()} / {len(dt_states_conf)} = {disagree.mean():.4f}")

# What states are involved in disagreements?
print(f"\nDisagreement matrix (DT state -> DFA state -> count):")
for dt_s in range(num_states):
    for dfa_s in range(num_states):
        count = ((dt_states_conf == dt_s) & (dfa_states_best == dfa_s) & disagree).sum()
        if count > 0:
            print(f"  DT={dt_s} -> DFA={dfa_s}: {count}")

# Check: are the DT-classified states correct according to the true oracle?
oracle = oracle_creator(SymmetricBernoulli(p_correct=1.0), 0)
print(f"\n=== Checking DT classifications against true oracle ===")
# For each confident prefix, what state does the true DFA assign?
confident_prefixes = [pre for pre, is_conf in zip(pst.prefixes, confident) if is_conf]
true_dfa_states = []
for pre in confident_prefixes:
    # Run the true DFA on this prefix
    state = outer.initial_state
    for sym in pre:
        state = outer.transitions[state][sym]
    true_dfa_states.append(state)
true_dfa_states = np.array(true_dfa_states)

print(f"Unique true states: {np.unique(true_dfa_states)}")
print(f"Unique DT states: {np.unique(dt_states_conf)}")
print(f"Unique learned DFA states: {np.unique(dfa_states_best)}")

# For each DT state, what true states are in it?
print(f"\nDT state -> true state distribution:")
for dt_s in range(num_states):
    mask = dt_states_conf == dt_s
    if mask.sum() == 0:
        continue
    unique, counts = np.unique(true_dfa_states[mask], return_counts=True)
    print(f"  DT state {dt_s} ({mask.sum()} prefixes): {dict(zip(unique.tolist(), counts.tolist()))}")

# Same for learned DFA
print(f"\nLearned DFA state -> true state distribution:")
for dfa_s in range(num_states):
    mask = dfa_states_best == dfa_s
    if mask.sum() == 0:
        continue
    unique, counts = np.unique(true_dfa_states[mask], return_counts=True)
    print(f"  DFA state {dfa_s} ({mask.sum()} prefixes): {dict(zip(unique.tolist(), counts.tolist()))}")

# On the disagreements specifically, who is right?
print(f"\n=== On disagreements: who is correct? ===")
disagree_dt = dt_states_conf[disagree]
disagree_dfa = dfa_states_best[disagree]
disagree_true = true_dfa_states[disagree]

# Map DT/DFA states to true states by majority vote
dt_to_true = {}
for dt_s in range(num_states):
    mask = dt_states_conf == dt_s
    if mask.sum() > 0:
        vals, counts = np.unique(true_dfa_states[mask], return_counts=True)
        dt_to_true[dt_s] = vals[np.argmax(counts)]

dfa_to_true = {}
for dfa_s in range(num_states):
    mask = dfa_states_best == dfa_s
    if mask.sum() > 0:
        vals, counts = np.unique(true_dfa_states[mask], return_counts=True)
        dfa_to_true[dfa_s] = vals[np.argmax(counts)]

print(f"DT state -> true state mapping: {dt_to_true}")
print(f"DFA state -> true state mapping: {dfa_to_true}")

dt_correct = sum(1 for i in range(len(disagree_dt)) if dt_to_true.get(disagree_dt[i]) == disagree_true[i])
dfa_correct = sum(1 for i in range(len(disagree_dfa)) if dfa_to_true.get(disagree_dfa[i]) == disagree_true[i])
print(f"On {disagree.sum()} disagreements: DT correct={dt_correct}, DFA correct={dfa_correct}, neither={disagree.sum()-dt_correct-dfa_correct}")
