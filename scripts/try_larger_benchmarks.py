"""Test L* on benchmarks of increasing size to find the frontier."""

import time

from orthogonal_dfa.l_star.examples.benchmark_generator import (
    DFAOracle,
    sample_balanced_benchmark,
)
from orthogonal_dfa.l_star.lstar import counterexample_driven_synthesis
from tests.test_lstar import compute_dfa_accuracy, compute_pst

TIME_LIMIT = 300  # seconds per benchmark


for num_states in [7, 10, 14, 18]:
    for seed in range(3):
        try:
            outer, inner, sep = sample_balanced_benchmark(
                seed,
                alphabet_size=2,
                num_inner_states=12,
                num_outer_states=num_states,
                probe_length=40,
                min_accept_or_reject=0.15,
            )
        except RuntimeError:
            print(f"states={num_states} seed={seed}: no benchmark found")
            continue

        oracle_creator = lambda nm, s, _dfa=outer: DFAOracle(nm, s, _dfa)
        pst = compute_pst(oracle_creator, 0.3, 0)

        t0 = time.time()
        last_dfa = None
        last_acc = 0
        for i, (dfa, dt, new_pst) in enumerate(
            counterexample_driven_synthesis(
                pst, additional_counterexamples=200, acc_threshold=0.98
            )
        ):
            elapsed = time.time() - t0
            acc, fp, fn = compute_dfa_accuracy(dfa, oracle_creator)
            last_dfa = dfa
            last_acc = acc
            if new_pst is not None:
                pst = new_pst
            if elapsed > TIME_LIMIT:
                break

        elapsed = time.time() - t0
        status = "PASS" if last_acc >= 0.95 else "FAIL"
        print(
            f"states={num_states} seed={seed}: {status} "
            f"learned={len(last_dfa.states)} acc={last_acc:.4f} "
            f"time={elapsed:.0f}s",
            flush=True,
        )
