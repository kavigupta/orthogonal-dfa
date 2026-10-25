"""Benchmark family 3: random minimal DFAs, filtered to E-L*'s regime.

The automata-lib DFA is the source of truth; `from_automata_dfa` ports it to the
upstream `capal.DFA` CAPAL learns.

Each target is drawn by `sample_minimal_admitted_dfa`, so E-L* is in regime on
all of them by construction, the same selection caveat as `our_targets`.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from orthogonal_dfa.capal_official import from_automata_dfa
from orthogonal_dfa.l_star.learn import DEFAULT_SAMPLE_LENGTH
from orthogonal_dfa.l_star.structures import NoisyOracle

from .benchmark import FAMILY_RANDOM, Benchmark

SIZES = [4, 8, 12]
COUNT = 100


def random_benchmarks(*, sizes: Sequence[int], count: int) -> List[Benchmark]:
    """``count`` targets of each size, named ``random_<size>_<index>``."""
    from orthogonal_dfa.l_star.examples.benchmark_generator import DFAOracle
    from orthogonal_dfa.l_star.examples.random_dfa import sample_minimal_admitted_dfa

    out: List[Benchmark] = []
    for size in sizes:
        for i in range(count):
            # Seeded per target, so a name always denotes the same DFA, whatever
            # the count: the sweep reuses cells by name.
            aut = sample_minimal_admitted_dfa(
                np.random.default_rng((size, i)),
                num_states=size,
                alphabet_size=2,
                length=DEFAULT_SAMPLE_LENGTH,
            )
            target = from_automata_dfa(aut)
            out.append(
                Benchmark(
                    name=f"random_{size:02d}_{i:03d}",
                    family=FAMILY_RANDOM,
                    target=target,
                    oracle_creator=lambda nm, s, _d=aut: NoisyOracle(
                        DFAOracle(_d), nm, s
                    ),
                    alphabet=list(target.alphabet),
                    symbols=len(target.alphabet),
                    target_states=size,
                )
            )
    return out
