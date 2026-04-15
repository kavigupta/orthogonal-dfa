"""Full convergence trace on the slow 11-state benchmark."""

import time

import numpy as np

from orthogonal_dfa.l_star.examples.benchmark_generator import (
    DFAOracle,
    sample_star_l_star,
)
from orthogonal_dfa.l_star.lstar import counterexample_driven_synthesis
from orthogonal_dfa.l_star.sampler import UniformSampler
from tests.test_lstar import compute_dfa_accuracy, compute_pst

us = UniformSampler(40)

probe_rng = np.random.default_rng(0)
for sub in range(10_000):
    rng = np.random.default_rng((0, sub))
    outer, inner, sep = sample_star_l_star(
        rng, num_inner_states=12, alphabet_size=2, num_accepting=1
    )
    n = len(outer.states)
    if not (3 <= n <= 12):
        continue
    rate = sum(outer.accepts_input(us.sample(probe_rng, 2)) for _ in range(200)) / 200
    if 0.05 < rate < 0.95:
        print(f"sub={sub} true_states={n} rate={rate:.3f}")
        break

oracle_creator = lambda nm, s: DFAOracle(nm, s, outer)
pst = compute_pst(oracle_creator, 0.4, 0)

t0 = time.time()
for i, (dfa, dt, new_pst) in enumerate(
    counterexample_driven_synthesis(
        pst, additional_counterexamples=200, acc_threshold=0.98
    )
):
    elapsed = time.time() - t0
    acc, fp, fn = compute_dfa_accuracy(dfa, oracle_creator)
    print(
        f"Round {i}: learned={len(dfa.states)} acc={acc:.4f} "
        f"fp={len(fp)} fn={len(fn)} "
        f"suffixes={len(pst.suffix_bank)} prefixes={len(pst.prefixes)} "
        f"elapsed={elapsed:.0f}s",
        flush=True,
    )
    if new_pst is not None:
        pst = new_pst

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.0f}s. Final: {len(dfa.states)} states, accuracy={acc:.4f}")
