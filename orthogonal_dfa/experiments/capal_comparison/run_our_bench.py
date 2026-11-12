#!/usr/bin/env python3
"""Experiment 2: both learners on this repo's benchmarks.

The modulo-9 and regex oracles from `tests/e2e/test_basic.py`, ported to upstream
`capal.DFA`s so CAPAL can learn the same languages.

    python -m orthogonal_dfa.experiments.capal_comparison.run_our_bench --processes 4
"""

from __future__ import annotations

import argparse

from .our_targets import OUR_BENCHMARK_NAMES, our_benchmarks
from .sweep import run_sweep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, required=True)
    args = parser.parse_args()
    run_sweep(
        OUR_BENCHMARK_NAMES,
        our_benchmarks,
        processes=args.processes,
        experiment="our_benchmarks",
        description=(
            "CAPAL and E-L* on this repo's oracle benchmarks (modulo-9 and the "
            "regex family from tests/e2e/test_basic.py) under persistent noise."
        ),
        generated_by="orthogonal_dfa.experiments.capal_comparison.run_our_bench",
    )


if __name__ == "__main__":
    main()
