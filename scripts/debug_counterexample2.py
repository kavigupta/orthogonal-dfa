"""Skip to round 2 and diagnose without running counterexample generation."""

import copy
import time
import numpy as np

from orthogonal_dfa.l_star.examples.benchmark_generator import (
    DFAOracle,
    sample_balanced_benchmark,
)
from orthogonal_dfa.l_star.lstar import (
    locate_incorrect_point,
    optimal_dfa,
    counterexample_driven_synthesis,
    add_counterexample_prefixes,
)
from orthogonal_dfa.l_star.state_discovery import discover_states
from orthogonal_dfa.l_star.structures import TriPredicate, SymmetricBernoulli
from orthogonal_dfa.l_star.statistics import compute_suffix_size_counterexample_gen as _compute_sfx
from tests.test_lstar import compute_dfa_accuracy, compute_pst
from orthogonal_dfa.l_star.dfa_utils import states_intermediate

outer, inner, sep = sample_balanced_benchmark(
    1, alphabet_size=2, num_inner_states=12, num_outer_states=10,
    probe_length=40, min_accept_or_reject=0.15,
)

oracle_creator = lambda nm, s, _dfa=outer: DFAOracle(nm, s, _dfa)
pst = compute_pst(oracle_creator, 0.3, 0)

# Manually do synthesis rounds without counterexample generation
# to get to the problematic state fast
print("=== Round 0: discover states ===")
t0 = time.time()
while True:
    dt = discover_states(pst, first_round=True)
    if dt.num_states > 1:
        break
    pst.sample_more_prefixes()
acc_rate, dfa = optimal_dfa(pst, dt)
print(f"DFA: {len(dfa.states)} states, acc_rate={acc_rate:.4f}, depth={dt.depth}")
acc, fp, fn = compute_dfa_accuracy(dfa, oracle_creator)
print(f"True acc={acc:.4f}, elapsed={time.time()-t0:.0f}s")

# Add counterexamples for round 0
add_counterexample_prefixes(pst, dt, dfa, 200)
print(f"Added 200 counterexamples, now {pst.num_prefixes} prefixes")

# Round 1
print("\n=== Round 1: discover states ===")
while True:
    dt = discover_states(pst, first_round=False)
    if dt.num_states > 1:
        break
    pst.sample_more_prefixes()
acc_rate, dfa = optimal_dfa(pst, dt)
print(f"DFA: {len(dfa.states)} states, acc_rate={acc_rate:.4f}, depth={dt.depth}")
acc, fp, fn = compute_dfa_accuracy(dfa, oracle_creator)
print(f"True acc={acc:.4f}, elapsed={time.time()-t0:.0f}s")

# Add counterexamples for round 1
add_counterexample_prefixes(pst, dt, dfa, 200)
print(f"Added 200 counterexamples, now {pst.num_prefixes} prefixes")

# Round 2 - this is the one that hangs
print("\n=== Round 2: discover states ===")
while True:
    dt = discover_states(pst, first_round=False)
    if dt.num_states > 1:
        break
    pst.sample_more_prefixes()
acc_rate, dfa = optimal_dfa(pst, dt)
print(f"DFA: {len(dfa.states)} states, acc_rate={acc_rate:.4f}, depth={dt.depth}")
acc, fp, fn = compute_dfa_accuracy(dfa, oracle_creator)
print(f"True acc={acc:.4f}, elapsed={time.time()-t0:.0f}s")

# Now diagnose THIS round
oracle = oracle_creator(SymmetricBernoulli(p_correct=1.0), 0)
boundary = pst.decision_boundary

string_len = pst.sampler.length
num_classifications = 2 + int(np.ceil(np.log2(string_len)))
num_node_decisions = num_classifications * dt.depth
effective_p = 0.5 + pst.config.min_signal_strength
per_node_budget = 0.2 / max(num_node_decisions, 1)
scaled_suffix_size = _compute_sfx(per_node_budget, effective_p)
print(f"\nscaled_suffix_size={scaled_suffix_size}, depth={dt.depth}, boundary={boundary:.4f}")
print(f"Full DT predicate sizes: {[len(n.predicate.vs) for n in _collect_internals(dt)]}" if False else "")

dt_reduced = dt.map_over_predicates(
    lambda p: TriPredicate(p.vs[:scaled_suffix_size], boundary, boundary)
)
dt_decisive = dt.map_over_predicates(
    lambda p: TriPredicate(p.vs, boundary, boundary)
)

us = pst.sampler
N = 50

print(f"\n=== Endpoint check ({N} trials) ===")
for label, tree in [("REDUCED", dt_reduced), ("DECISIVE", dt_decisive)]:
    agrees = disagrees = nones = 0
    for trial in range(N):
        y = us.sample(pst.rng, pst.alphabet_size)
        s0 = tree.classify([], oracle)
        dfa_states = states_intermediate(s0, y, dfa)
        dt_end = tree.classify(y, oracle)
        if dt_end is None:
            nones += 1
        elif dt_end == dfa_states[-1]:
            agrees += 1
        else:
            disagrees += 1
    print(f"  {label}: agree={agrees} disagree={disagrees} none={nones}")

# Now try locate_incorrect_point on a few samples
print(f"\n=== Full pipeline check (20 trials) ===")
found = 0
locate_none = 0
decisive_filter = 0
for trial in range(20):
    y = us.sample(pst.rng, pst.alphabet_size)
    result = locate_incorrect_point(oracle, dt_reduced, dfa, [], y)
    if result is None:
        locate_none += 1
        continue
    prefix, sym = result
    state_1 = dt_decisive.classify(prefix, oracle)
    if state_1 is None:
        decisive_filter += 1
        continue
    state_2 = dfa.transitions[state_1][sym]
    state_after = dt_decisive.classify(prefix + [sym], oracle)
    if state_after is None:
        decisive_filter += 1
    elif state_2 == state_after:
        decisive_filter += 1
    else:
        found += 1

print(f"  locate_none={locate_none} decisive_filter={decisive_filter} found={found}")

elapsed = time.time() - t0
print(f"\nTotal elapsed: {elapsed:.0f}s")
