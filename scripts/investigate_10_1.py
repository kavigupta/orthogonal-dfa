"""Test empty-x fix on 10-state seed=1."""

import time

from orthogonal_dfa.l_star.examples.benchmark_generator import (
    DFAOracle,
    sample_balanced_benchmark,
)
from orthogonal_dfa.l_star.lstar import counterexample_driven_synthesis
from tests.test_lstar import compute_dfa_accuracy, compute_pst

outer, inner, sep = sample_balanced_benchmark(
    1,
    alphabet_size=2,
    num_inner_states=12,
    num_outer_states=10,
    probe_length=40,
    min_accept_or_reject=0.15,
)
print(f"outer={len(outer.states)}", flush=True)

oracle_creator = lambda nm, s, _dfa=outer: DFAOracle(nm, s, _dfa)
pst = compute_pst(oracle_creator, 0.3, 0)

t0 = time.time()
for i, (dfa, dt, new_pst) in enumerate(
    counterexample_driven_synthesis(
        pst, additional_counterexamples=200, acc_threshold=0.98
    )
):
    elapsed = time.time() - t0
    acc, fp, fn = compute_dfa_accuracy(dfa, oracle_creator)
    print(
        f"round {i}: learned={len(dfa.states)} acc={acc:.4f} "
        f"fp={len(fp)} fn={len(fn)} elapsed={elapsed:.0f}s",
        flush=True,
    )
    if new_pst is not None:
        pst = new_pst
    if elapsed > 300:
        print("TIMEOUT", flush=True)
        break

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.0f}s", flush=True)
