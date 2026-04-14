"""Debug why counterexample generation hangs on later rounds of 10-state seed=1."""

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
)
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

# Run synthesis rounds, diagnose every round
t0 = time.time()
for i, (dfa, dt, new_pst) in enumerate(
    counterexample_driven_synthesis(
        pst, additional_counterexamples=200, acc_threshold=0.98
    )
):
    elapsed = time.time() - t0
    acc, fp, fn = compute_dfa_accuracy(dfa, oracle_creator)
    print(f"\n{'='*60}")
    print(f"round {i}: learned={len(dfa.states)} acc={acc:.4f} dt_depth={dt.depth} elapsed={elapsed:.0f}s")

    oracle = oracle_creator(SymmetricBernoulli(p_correct=1.0), 0)
    boundary = pst.decision_boundary

    string_len = pst.sampler.length
    num_classifications = 2 + int(np.ceil(np.log2(string_len)))
    num_node_decisions = num_classifications * dt.depth
    effective_p = 0.5 + pst.config.min_signal_strength
    per_node_budget = 0.2 / max(num_node_decisions, 1)
    scaled_suffix_size = _compute_sfx(per_node_budget, effective_p)
    print(f"  suffix_size={scaled_suffix_size}, depth={dt.depth}, boundary={boundary:.4f}")

    dt_reduced = dt.map_over_predicates(
        lambda p: TriPredicate(p.vs[:scaled_suffix_size], boundary, boundary)
    )
    dt_decisive = dt.map_over_predicates(
        lambda p: TriPredicate(p.vs, boundary, boundary)
    )

    us = pst.sampler
    N = 50

    # Fast check: just endpoint agreement with reduced DT
    endpoint_agrees = 0
    endpoint_disagrees = 0
    reduced_none = 0
    for trial in range(N):
        y = us.sample(pst.rng, pst.alphabet_size)
        s0 = dt_reduced.classify([], oracle)
        dfa_states = states_intermediate(s0, y, dfa)
        dt_end = dt_reduced.classify(y, oracle)
        if dt_end is None:
            reduced_none += 1
        elif dt_end == dfa_states[-1]:
            endpoint_agrees += 1
        else:
            endpoint_disagrees += 1

    print(f"  REDUCED endpoint ({N} trials): agree={endpoint_agrees} disagree={endpoint_disagrees} none={reduced_none}")

    # Check with decisive DT
    d_agrees = 0
    d_disagrees = 0
    d_none = 0
    for trial in range(N):
        y = us.sample(pst.rng, pst.alphabet_size)
        s0 = dt_decisive.classify([], oracle)
        dfa_states = states_intermediate(s0, y, dfa)
        dt_end = dt_decisive.classify(y, oracle)
        if dt_end is None:
            d_none += 1
        elif dt_end == dfa_states[-1]:
            d_agrees += 1
        else:
            d_disagrees += 1

    print(f"  DECISIVE endpoint ({N} trials): agree={d_agrees} disagree={d_disagrees} none={d_none}")

    if new_pst is not None:
        pst = new_pst
    if elapsed > 250:
        print("TIMEOUT")
        break
