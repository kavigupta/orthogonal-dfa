#!/usr/bin/env python3
"""Experiment 3: both learners on random minimal DFAs of a few sizes.

    python -m orthogonal_dfa.experiments.capal_comparison.run_random_bench
"""

from __future__ import annotations

from .random_targets import COUNT, SIZES, random_benchmarks
from .sweep import run_sweep


def main() -> None:
    run_sweep(
        random_benchmarks(sizes=SIZES, count=COUNT),
        experiment="random_benchmarks",
        description=(
            f"CAPAL and E-L* on {COUNT} random minimal DFAs of each size in "
            f"{SIZES}, drawn to satisfy E-L*'s preconditions, under persistent noise."
        ),
        generated_by="orthogonal_dfa.experiments.capal_comparison.run_random_bench",
    )


if __name__ == "__main__":
    main()
