#!/usr/bin/env python3
"""Experiment 2: both learners on this repo's benchmarks.

The modulo-9 and regex oracles from `tests/test_lstar.py`, ported to upstream
`capal.DFA`s so CAPAL can learn the same languages.

    python -m orthogonal_dfa.experiments.capal_comparison.run_our_bench
"""

from __future__ import annotations

from .our_targets import our_benchmarks
from .sweep import run_sweep


def main() -> None:
    run_sweep(
        our_benchmarks(),
        experiment="our_benchmarks",
        description=(
            "CAPAL and E-L* on this repo's oracle benchmarks (modulo-9 and the "
            "regex family from tests/test_lstar.py) under persistent noise."
        ),
        generated_by="orthogonal_dfa.experiments.capal_comparison.run_our_bench",
    )


if __name__ == "__main__":
    main()
