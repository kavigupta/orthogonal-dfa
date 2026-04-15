"""Check what benchmarks are generated with the new parameters."""

import numpy as np

from orthogonal_dfa.l_star.examples.benchmark_generator import sample_balanced_benchmark

for seed in range(3):
    outer, inner, sep = sample_balanced_benchmark(
        seed,
        alphabet_size=2,
        num_inner_states=12,
        num_outer_states=7,
        probe_length=40,
        min_accept_or_reject=0.15,
    )
    from orthogonal_dfa.l_star.sampler import UniformSampler

    us = UniformSampler(40)
    rng = np.random.default_rng(seed)
    rate = sum(outer.accepts_input(us.sample(rng, 2)) for _ in range(500)) / 500
    print(
        f"seed={seed} outer_states={len(outer.states)} "
        f"inner_states={len(inner.states)} rate={rate:.3f}"
    )
